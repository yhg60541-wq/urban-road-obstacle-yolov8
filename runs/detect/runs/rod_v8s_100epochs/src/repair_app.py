import gradio as gr
from datetime import datetime
import json
import os
import shutil
import uuid
from PIL import Image
import numpy as np

SHARED_DATA_FILE = 'shared_reports.json'
UPLOAD_DIR = os.path.abspath('uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def safe_path(p):
    """确保返回给 Gradio 的路径是有效的文件路径"""
    if p and isinstance(p, str) and os.path.isfile(p):
        return p
    return None


def load_reports():
    if os.path.exists(SHARED_DATA_FILE):
        try:
            with open(SHARED_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('reports', []), data.get('user_points', {})
        except Exception:
            return [], {}
    return [], {}


def save_reports(report_list, point_dict):
    save_dict = {"reports": report_list, "user_points": point_dict, "last_update": datetime.now().isoformat()}
    with open(SHARED_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)


def save_image(img_input, prefix='repair'):
    if img_input is None:
        return None
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    fpath = os.path.join(UPLOAD_DIR, fname)
    try:
        # 字符串路径（Gradio Gallery/File 上传返回的路径）
        if isinstance(img_input, str):
            if os.path.isfile(img_input):
                shutil.copy(img_input, fpath)
                return fpath
            return None
        # PIL Image
        if isinstance(img_input, Image.Image):
            img_input.save(fpath)
            return fpath
        # numpy 数组
        if isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                if img_input.max() <= 1.0:
                    img_input = (img_input * 255).astype(np.uint8)
                else:
                    img_input = img_input.astype(np.uint8)
            Image.fromarray(img_input).save(fpath)
            return fpath
        # 列表/元组（Gallery 可能返回列表）
        if isinstance(img_input, (list, tuple)):
            if len(img_input) == 0:
                return None
            return save_image(img_input[0], prefix)
        print(f"未知图片类型: {type(img_input)}")
        return None
    except Exception as e:
        print(f"保存图片失败: {e}")
        return None


def get_team_tasks(team_name, status_filter="全部"):
    rep, _ = load_reports()
    tasks = []
    for r in rep:
        repair = r.get('repair', {})
        if repair.get('team') == team_name or r.get('assignee') == team_name:
            if status_filter == "全部" or r['status'] == status_filter:
                tasks.append(r)
    return tasks


def get_report_choices(tasks):
    choices = []
    for r in tasks:
        status_emoji = {"待审核": "🟡", "审核通过": "🟣", "处理中": "🔵", "已整改": "🟢", "已驳回": "🔴"}.get(r['status'], "⚪")
        objs = ", ".join([f"{o['class_zh']}" for o in r['objects'][:2]])
        if len(r['objects']) > 2:
            objs += "..."
        label = f"{r['id']} | {status_emoji}{r['status']} | 📍{r['location'][:15]}... | {objs[:15]}"
        choices.append((label, r['id']))
    return choices


def parse_choice(choice):
    if not choice:
        return ""
    if " | " in choice:
        return choice.split(" | ")[0]
    return choice.strip()


def get_task_detail(choice):
    rid = parse_choice(choice)
    if not rid:
        return "请输入工单编号", None, None, None, None, "", "", "", ""
    rep, _ = load_reports()
    for r in rep:
        if r['id'] == rid:
            lines = [
                f"单号: {r['id']}",
                f"上报时间: {r['timestamp'][:19]}",
                f"位置: {r['location']}",
                f"上报人: {r['reporter']}",
                f"当前状态: {r['status']}",
                "异常清单:"
            ]
            for o in r['objects']:
                lines.append(f"  - {o['class_zh']}: {o['abnormal']}")
            if r.get('notes'):
                lines.append(f"补充说明: {r['notes']}")
            repair = r.get('repair', {})
            status = repair.get('repair_status', '待接单')
            notes = repair.get('repair_notes', '')
            return (
                "\n".join(lines),
                safe_path(r.get('original_image')),
                safe_path(r.get('annotated_image')),
                [safe_path(p) for p in repair.get('repair_photos', []) if safe_path(p)],
                [safe_path(p) for p in repair.get('after_photos', []) if safe_path(p)],
                status,
                notes,
                r.get('location', ''),
                r.get('status', '')
            )
    return "未找到该工单", None, None, None, None, "", "", "", ""


def accept_task(choice, team_name):
    rid = parse_choice(choice)
    if not rid:
        return "❌ 请输入工单编号", ""
    rep, pts = load_reports()
    for r in rep:
        if r['id'] == rid:
            if r.get('status') not in ['审核通过', '处理中']:
                return f"❌ 工单{r['id']}当前状态为{r['status']}，无法接单", ""
            r['status'] = '处理中'
            repair = r.get('repair', {})
            repair['team'] = team_name
            repair['repair_status'] = '维修中'
            repair['accept_time'] = datetime.now().isoformat()
            r['repair'] = repair
            r['update_time'] = datetime.now().isoformat()
            save_reports(rep, pts)
            tasks = get_team_tasks(team_name, "全部")
            return f"✅ 已接单：{rid}", get_team_list_html(tasks)
    return "❌ 未找到工单", ""


def submit_repair(choice, team_name, repair_note, repair_imgs, after_imgs):
    rid = parse_choice(choice)
    if not rid:
        return "❌ 请输入工单编号", "", None, None, None, "", "", "", ""
    rep, pts = load_reports()
    for r in rep:
        if r['id'] == rid:
            repair = r.get('repair', {})
            if repair.get('team') != team_name:
                return "❌ 该工单未分配给您", "", None, None, None, "", "", "", ""
            rp = repair.get('repair_photos', [])
            if repair_imgs:
                for img in repair_imgs:
                    p = save_image(img, 'repair_process')
                    if p:
                        rp.append(p)
            ap = repair.get('after_photos', [])
            if after_imgs:
                for img in after_imgs:
                    p = save_image(img, 'repair_after')
                    if p:
                        ap.append(p)
            repair['repair_photos'] = rp
            repair['after_photos'] = ap
            repair['repair_notes'] = repair_note.strip() if repair_note else repair.get('repair_notes', '')
            repair['repair_status'] = '待验收'
            repair['complete_time'] = datetime.now().isoformat()
            r['repair'] = repair
            r['status'] = '已整改'
            r['update_time'] = datetime.now().isoformat()
            save_reports(rep, pts)
            return (
                f"✅ 工单{rid}已提交验收，等待管理员确认",
                "\n".join([f"单号: {r['id']}", f"状态: 已整改", f"维修备注: {repair['repair_notes']}"]),
                safe_path(r.get('original_image')),
                safe_path(r.get('annotated_image')),
                [safe_path(p) for p in rp if safe_path(p)],
                [safe_path(p) for p in ap if safe_path(p)],
                '待验收',
                repair['repair_notes'],
                r.get('location', ''),
                r.get('status', '')
            )
    return "❌ 未找到工单", "", None, None, None, "", "", "", ""


def get_team_list_html(tasks):
    if not tasks:
        return '<div style="text-align:center; padding:50px; color:#999">暂无任务</div>'
    color_map = {
        "待审核": "#ffc107", "审核通过": "#6f42c1", "处理中": "#17a2b8",
        "已整改": "#28a745", "已驳回": "#dc3545"
    }
    html = '<div style="max-height:500px; overflow-y:auto">'
    for r in tasks[:30]:
        s = r['status']
        c = color_map.get(s, "#888")
        repair = r.get('repair', {})
        r_status = repair.get('repair_status', '')
        r_badge = f'<span style="background:#fd7e14; color:white; padding:2px 8px; border-radius:10px; font-size:11px; margin-left:5px">{r_status}</span>' if r_status else ''
        obj_str = ""
        for o in r["objects"][:3]:
            obj_str += f'<span style="background:#e9ecef; padding:2px 6px; border-radius:8px; font-size:11px; margin:2px">{o["class_zh"]}</span>'
        html += (
            f'<div style="border:1px solid #ddd; border-radius:10px; margin:10px; padding:12px; background:white">'
            f'<div style="display:flex; justify-content:space-between; align-items:center">'
            f'<div><b>📋 {r["id"]}</b> <span style="background:{c}; color:white; padding:2px 8px; border-radius:12px; font-size:12px">{s}</span>{r_badge}</div>'
            f'<div style="font-size:12px; color:#666">📅 {r["timestamp"][:19]}</div>'
            f'</div>'
            f'<div style="margin-top:6px; font-size:13px">📍 {r.get("location","未知")}</div>'
            f'<div style="margin-top:4px">{obj_str}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def get_history_html(team_name):
    rep, _ = load_reports()
    history = []
    for r in rep:
        repair = r.get('repair', {})
        if repair.get('team') == team_name and r['status'] == '已整改':
            history.append(r)
    if not history:
        return '<div style="text-align:center; padding:50px; color:#999">暂无历史记录</div>'
    html = '<div style="max-height:500px; overflow-y:auto">'
    for r in history[:20]:
        repair = r.get('repair', {})
        after_count = len(repair.get('after_photos', []))
        html += (
            f'<div style="border:1px solid #28a745; border-radius:10px; margin:10px; padding:12px; background:#f8fff8">'
            f'<div style="display:flex; justify-content:space-between">'
            f'<div><b>📋 {r["id"]}</b> <span style="background:#28a745; color:white; padding:2px 8px; border-radius:12px; font-size:12px">已完成</span></div>'
            f'<div style="font-size:12px; color:#666">{repair.get("complete_time","")[:19]}</div>'
            f'</div>'
            f'<div style="margin-top:6px; font-size:13px">📍 {r.get("location","未知")}</div>'
            f'<div style="margin-top:4px; font-size:12px; color:#666">整改照片: {after_count}张 | 备注: {repair.get("repair_notes","无")}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


with gr.Blocks(title="市政维修端", theme=gr.themes.Soft(), css="footer{visibility:hidden}") as demo:
    gr.HTML(
        '<div style="background:linear-gradient(135deg,#fd7e14,#e8590c);padding:20px;border-radius:16px;color:white;text-align:center">'
        '<h1>🔧 市政维修作业端</h1><p>接单 · 处理 · 反馈 · 闭环</p></div>'
    )

    team_input = gr.Dropdown(
        choices=['城厢区维修队', '荔城区维修队', '涵江区维修队', '秀屿区维修队', '市政工程队'],
        label="选择维修队伍",
        value='城厢区维修队'
    )

    with gr.Tabs():
        with gr.TabItem("📥 任务大厅"):
            with gr.Row():
                task_filter = gr.Dropdown(
                    choices=["全部", "审核通过", "处理中", "已整改"],
                    value="全部", label="任务状态筛选"
                )
                refresh_tasks = gr.Button("🔄 刷新任务列表")
            task_list_html = gr.HTML(value=get_team_list_html([]))

            gr.Markdown("### 📂 任务处理")
            with gr.Row():
                task_id_input = gr.Dropdown(
                    choices=[],
                    label="选择工单查看详情",
                    allow_custom_value=True,
                    scale=3,
                    info="选中后自动加载图片和详情"
                )
                accept_task_btn = gr.Button("🚀 接单", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    task_detail = gr.Textbox(label="任务详情", lines=12, interactive=False)
                    task_location = gr.Textbox(label="📍 位置", interactive=False)
                    task_sys_status = gr.Textbox(label="系统状态", interactive=False)
                    task_repair_status = gr.Textbox(label="维修状态", interactive=False)
                with gr.Column():
                    task_orig_img = gr.Image(label="📷 市民上报原图", type="filepath", height=220)
                    task_ann_img = gr.Image(label="🤖 AI识别图", type="filepath", height=220)

            task_msg = gr.Textbox(label="操作反馈", interactive=False)

        with gr.TabItem("🔨 维修处理"):
            gr.Markdown("### 提交维修结果（接单后才能处理）")
            with gr.Row():
                work_id_input = gr.Dropdown(
                    choices=[],
                    label="选择工单",
                    allow_custom_value=True,
                    scale=3,
                    info="选中后自动加载任务信息"
                )
            with gr.Row():
                with gr.Column():
                    work_detail = gr.Textbox(label="当前任务信息", lines=8, interactive=False)
                    work_repair_note = gr.Textbox(label="📝 维修备注", placeholder="填写维修过程、更换部件、处理结果等...", lines=3)
                with gr.Column():
                    work_repair_imgs = gr.File(label="📷 维修过程照片（可选）", file_count="multiple")
                    work_after_imgs = gr.File(label="📸 整改后照片（必填，至少一张）", file_count="multiple")
            submit_repair_btn = gr.Button("✅ 提交验收", variant="primary")
            work_result = gr.Textbox(label="提交结果", interactive=False)

            with gr.Row():
                with gr.Column():
                    res_orig = gr.Image(label="原图", type="filepath", height=180)
                with gr.Column():
                    res_repair = gr.Gallery(label="维修过程图", columns=2, height=180)
                with gr.Column():
                    res_after = gr.Gallery(label="整改后照片", columns=2, height=180)

        with gr.TabItem("📜 历史任务"):
            with gr.Row():
                refresh_history = gr.Button("🔄 刷新历史")
            history_html = gr.HTML(value=get_history_html('城厢区维修队'))

    def refresh_tasks_fn(team, st):
        tasks = get_team_tasks(team, st)
        ids = get_report_choices(tasks)
        return get_team_list_html(tasks), gr.update(choices=ids), gr.update(choices=ids)

    refresh_tasks.click(
        refresh_tasks_fn,
        [team_input, task_filter],
        [task_list_html, task_id_input, work_id_input]
    )

    def on_task_select_change(choice):
        return get_task_detail(choice)
    task_id_input.change(
        on_task_select_change,
        [task_id_input],
        [task_detail, task_orig_img, task_ann_img, gr.State(), gr.State(), task_repair_status, gr.State(), task_location, task_sys_status]
    )

    def accept_fn(choice, team):
        msg, html = accept_task(choice, team)
        tasks = get_team_tasks(team, "全部")
        ids = get_report_choices(tasks)
        return msg, html, gr.update(choices=ids), gr.update(choices=ids)

    accept_task_btn.click(
        accept_fn,
        [task_id_input, team_input],
        [task_msg, task_list_html, task_id_input, work_id_input]
    )

    def on_work_select_change(choice):
        detail, orig, ann, rp, ap, rs, rn, loc, ss = get_task_detail(choice)
        return detail, gr.update(value=rp), gr.update(value=ap), rs, rn, orig, gr.update(value=rp), gr.update(value=ap)

    work_id_input.change(
        on_work_select_change,
        [work_id_input],
        [work_detail, work_repair_imgs, work_after_imgs, gr.State(), work_repair_note, res_orig, res_repair, res_after]
    )

    def submit_fn(choice, team, note, r_imgs, a_imgs):
        if not parse_choice(choice):
            return "❌ 请选择工单", "", None, None, None, "", "", "", ""
        if not a_imgs or len(a_imgs) == 0:
            return "❌ 请至少上传一张整改后照片", "", None, None, None, "", "", "", ""
        msg, detail, orig, ann, rp, ap, rs, rn, loc, ss = submit_repair(choice, team, note, r_imgs, a_imgs)
        return msg, detail, orig, gr.update(value=rp), gr.update(value=ap), rs, rn, loc, ss

    submit_repair_btn.click(
        submit_fn,
        [work_id_input, team_input, work_repair_note, work_repair_imgs, work_after_imgs],
        [work_result, work_detail, res_orig, res_repair, res_after, gr.State(), gr.State(), gr.State(), gr.State()]
    )

    def refresh_hist_fn(team):
        return get_history_html(team)
    refresh_history.click(refresh_hist_fn, [team_input], [history_html])

    demo.load(
        lambda team: refresh_tasks_fn(team, "全部"),
        [team_input],
        [task_list_html, task_id_input, work_id_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8082, share=False)