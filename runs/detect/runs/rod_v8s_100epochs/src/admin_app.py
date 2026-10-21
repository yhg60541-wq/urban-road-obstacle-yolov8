import gradio as gr
from datetime import datetime
import json
import os
import pandas as pd
from io import BytesIO
import shutil

SHARED_DATA_FILE = 'shared_reports.json'
ADMIN_CONFIG_FILE = 'admin_config.json'
BACKUP_DIR = 'backups'
DEFAULT_CONFIG = {
    'announcements': ['🚧 荔城中大道井盖维修施工中', '⚠️ 注意路面坑洞及时上报'],
    'users': {'admin': {'password': 'admin123', 'role': '超级管理员'}, 'operator': {'password': 'operator123', 'role': '操作员'}},
    'repair_teams': ['城厢区维修队', '荔城区维修队', '涵江区维修队', '秀屿区维修队', '市政工程队']
}
os.makedirs(BACKUP_DIR, exist_ok=True)


def safe_path(p):
    """确保返回给 Gradio 的路径是有效的文件路径"""
    if p and isinstance(p, str) and os.path.isfile(p):
        return p
    return None


def save_config(cfg):
    with open(ADMIN_CONFIG_FILE, 'w', encoding='utf-8') as fp:
        json.dump(cfg, fp=fp, ensure_ascii=False, indent=2)


def load_config():
    if os.path.exists(ADMIN_CONFIG_FILE):
        try:
            with open(ADMIN_CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    if k not in cfg:
                        cfg[k] = v
                save_config(cfg)
                return cfg
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


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
    if os.path.exists(SHARED_DATA_FILE):
        bak_path = os.path.join(BACKUP_DIR, f"bk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(SHARED_DATA_FILE, 'r', encoding='utf-8') as fr, open(bak_path, 'w', encoding='utf-8') as fw:
                json.dump(json.load(fr), fw, ensure_ascii=False, indent=2)
        except Exception:
            pass
    save_dict = {"reports": report_list, "user_points": point_dict, "last_update": datetime.now().isoformat()}
    with open(SHARED_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)


def get_report_choices(status_filter=None):
    rep, _ = load_reports()
    choices = []
    for r in rep:
        if status_filter and r['status'] != status_filter:
            continue
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


def change_report_status(rid, new_status, assignee="", feedback="", reject_note=""):
    if not rid or not rid.strip():
        return "❌ 请输入工单编号", get_reports_html()
    rep_list, pts = load_reports()
    flag = False
    for item in rep_list:
        if item['id'] == rid:
            item['status'] = new_status
            item['update_time'] = datetime.now().isoformat()
            if assignee:
                item['assignee'] = assignee
            if feedback:
                item['admin_feedback'] = feedback
            if reject_note:
                item['reject_reason'] = reject_note
            if new_status == '处理中' and assignee:
                repair = item.get('repair', {})
                repair['team'] = assignee
                item['repair'] = repair
            flag = True
            break
    if flag:
        save_reports(rep_list, pts)
        return f"✅ {rid} 操作成功，状态更新为：{new_status}", get_reports_html()
    return f"❌ 未找到工单{rid}", get_reports_html()


def delete_single(rid):
    if not rid or not rid.strip():
        return "❌ 请输入工单编号", get_reports_html()
    rep_list, pts = load_reports()
    old_len = len(rep_list)
    rep_list = [item for item in rep_list if item['id'] != rid]
    if len(rep_list) < old_len:
        save_reports(rep_list, pts)
        return f"✅ 删除{rid}成功", get_reports_html()
    return f"❌ 无此工单", get_reports_html()


def audit_report(rid, result, note=""):
    if not rid or not rid.strip():
        return "❌ 请输入工单编号", get_reports_html(), None, None, ""
    rep_list, pts = load_reports()
    for item in rep_list:
        if item['id'] == rid:
            if result == '通过':
                item['status'] = '审核通过'
                item['audit_info'] = {
                    'auditor': '管理员',
                    'audit_time': datetime.now().isoformat(),
                    'result': '通过',
                    'note': note
                }
            else:
                item['status'] = '已驳回'
                item['reject_reason'] = note
                item['audit_info'] = {
                    'auditor': '管理员',
                    'audit_time': datetime.now().isoformat(),
                    'result': '驳回',
                    'note': note
                }
            item['update_time'] = datetime.now().isoformat()
            save_reports(rep_list, pts)
            orig = safe_path(item.get('original_image'))
            ann = safe_path(item.get('annotated_image'))
            detail = build_detail_text(item)
            return f"✅ {rid} 审核{result}成功", get_reports_html(), orig, ann, detail
    return f"❌ 未找到工单{rid}", get_reports_html(), None, None, ""


def confirm_repair_fn(rid, confirm_note):
    if not rid or not rid.strip():
        return "❌ 请输入工单编号", "", None, None, None, None, ""
    rep_list, pts = load_reports()
    for item in rep_list:
        if item['id'] == rid:
            item['status'] = '已整改'
            item['admin_feedback'] = confirm_note if confirm_note else '管理员已确认维修完成'
            item['update_time'] = datetime.now().isoformat()
            repair = item.get('repair', {})
            repair['repair_status'] = '已验收'
            item['repair'] = repair
            save_reports(rep_list, pts)
            detail = build_detail_text(item)
            orig = safe_path(item.get('original_image'))
            ann = safe_path(item.get('annotated_image'))
            repair_photos = repair.get('repair_photos', [])
            after_photos = repair.get('after_photos', [])
            rp_first = safe_path(repair_photos[0]) if repair_photos else None
            af_first = safe_path(after_photos[0]) if after_photos else None
            return (
                f"✅ {rid} 已确认验收通过",
                detail,
                orig,
                ann,
                rp_first,
                af_first,
                get_reports_html()
            )
    return "❌ 未找到工单", "", None, None, None, None, ""


def build_detail_text(r):
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
    if r.get('assignee'):
        lines.append(f"分派队伍: {r['assignee']}")
    if r.get('admin_feedback'):
        lines.append(f"管理员备注: {r['admin_feedback']}")
    if r.get('reject_reason'):
        lines.append(f"驳回原因: {r['reject_reason']}")
    audit = r.get('audit_info', {})
    if audit:
        lines.append(f"审核结果: {audit.get('result','')} ({audit.get('audit_time','')[:19]})")
    repair = r.get('repair', {})
    if repair.get('team'):
        lines.append(f"维修队伍: {repair['team']}")
    if repair.get('accept_time'):
        lines.append(f"接单时间: {repair['accept_time'][:19]}")
    if repair.get('complete_time'):
        lines.append(f"完成时间: {repair['complete_time'][:19]}")
    if repair.get('repair_notes'):
        lines.append(f"维修备注: {repair['repair_notes']}")
    if repair.get('repair_status'):
        lines.append(f"维修状态: {repair['repair_status']}")
    after_count = len(repair.get('after_photos', []))
    repair_count = len(repair.get('repair_photos', []))
    lines.append(f"维修照片: {repair_count}张 | 整改照片: {after_count}张")
    return "\n".join(lines)


def get_detail_data(choice):
    rid = parse_choice(choice)
    if not rid:
        return "请选择工单", None, None, None, None, "", "", "", ""
    rep_list, _ = load_reports()
    for r in rep_list:
        if r['id'] == rid:
            detail = build_detail_text(r)
            repair = r.get('repair', {})
            repair_photos = repair.get('repair_photos', [])
            after_photos = repair.get('after_photos', [])
            rp_first = safe_path(repair_photos[0]) if repair_photos else None
            af_first = safe_path(after_photos[0]) if after_photos else None
            return (
                detail,
                safe_path(r.get('original_image')),
                safe_path(r.get('annotated_image')),
                rp_first,
                af_first,
                r.get('status', ''),
                r.get('location', ''),
                repair.get('repair_notes', ''),
                repair.get('repair_status', '')
            )
    return "找不到该工单", None, None, None, None, "", "", "", ""


def dashboard_html():
    rep, _ = load_reports()
    total = len(rep)
    d_pending = len([x for x in rep if x['status'] == "待审核"])
    d_audited = len([x for x in rep if x['status'] == "审核通过"])
    d_proc = len([x for x in rep if x['status'] == "处理中"])
    d_ok = len([x for x in rep if x['status'] == "已整改"])
    d_refuse = len([x for x in rep if x['status'] == "已驳回"])
    cls_cnt = {}
    for i in rep:
        for o in i['objects']:
            k = o['class_zh']
            cls_cnt[k] = cls_cnt.get(k, 0) + 1
    cls_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:15px">'
    for k, v in sorted(cls_cnt.items(), key=lambda x: x[1], reverse=True)[:10]:
        cls_html += f'<div style="background:#eee;padding:8px 15px;border-radius:8px">{k}:{v}单</div>'
    cls_html += "</div>"
    html = (
        f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px">'
        f'<div style="background:#5558dd;color:white;padding:15px;border-radius:12px;text-align:center"><div style="font-size:28px">{total}</div>总工单</div>'
        f'<div style="background:#ffc107;padding:15px;border-radius:12px;text-align:center"><div style="font-size:28px">{d_pending}</div>待审核</div>'
        f'<div style="background:#6f42c1;color:white;padding:15px;border-radius:12px;text-align:center"><div style="font-size:28px">{d_audited}</div>审核通过</div>'
        f'<div style="background:#17a2b8;color:white;padding:15px;border-radius:12px;text-align:center"><div style="font-size:28px">{d_proc}</div>处理中</div>'
        f'<div style="background:#28a745;color:white;padding:15px;border-radius:12px;text-align:center"><div style="font-size:28px">{d_ok}</div>已整改</div>'
        f'<div style="background:#dc3545;color:white;padding:15px;text-align:center;border-radius:12px"><div style="font-size:28px">{d_refuse}</div>已驳回</div>'
        f'</div>'
        f'<h3>故障分类TOP10</h3>{cls_html}'
    )
    return html


def get_reports_html(search="", filter_status="全部"):
    rep, _ = load_reports()
    if filter_status != "全部":
        rep = [x for x in rep if x['status'] == filter_status]
    if search.strip():
        rep = [x for x in rep if search in x['id'] or search in x['location'] or search in x['reporter']]
    if not rep:
        return "<div style='padding:40px;text-align:center;color:#999'>暂无工单数据</div>"
    color_map = {
        "待审核": "#ffc107", "审核通过": "#6f42c1", "处理中": "#17a2b8",
        "已整改": "#28a745", "已驳回": "#dc3545"
    }
    html = '<div style="max-height:620px;overflow-y:auto">'
    for item in rep[:50]:
        s = item['status']
        c = color_map.get(s, "#888")
        obj_str = ""
        for o in item['objects']:
            obj_str += f'<span style="background:#eee;padding:2px 6px;border-radius:10px;margin:2px;font-size:11px">{o["class_zh"]}:{o["abnormal"]}</span>'
        ass = item.get("assignee", "未分派")
        has_img = '✅' if item.get('original_image') else '❌'
        repair = item.get('repair', {})
        has_after = '✅' if repair.get('after_photos') else '❌'
        html += (
            f'<div style="border:1px solid #ddd;border-radius:10px;padding:12px;margin:8px">'
            f'<div style="display:flex;justify-content:space-between">'
            f'<span><b>{item["id"]}</b> <span style="background:{c};color:white;padding:2px 8px;border-radius:12px;font-size:12px">{s}</span></span>'
            f'<span style="font-size:12px;color:#777">{item["timestamp"][:19]} 📷原图{has_img} 📸整改{has_after}</span>'
            f'</div>'
            f'<div style="margin:6px 0;font-size:13px">📍{item["location"]} | 上报人:{item["reporter"]} | 维修队:{ass}</div>'
            f'<div>{obj_str}</div>'
            f'</div>'
        )
    html += "</div>"
    return html


def add_ann(text):
    cfg = load_config()
    if not text.strip():
        return "❌内容不能为空", get_ann_html()
    cfg['announcements'].insert(0, text.strip())
    if len(cfg['announcements']) > 10:
        cfg['announcements'] = cfg['announcements'][:10]
    save_config(cfg)
    return "✅公告添加成功", get_ann_html()


def get_ann_html():
    cfg = load_config()
    ann = cfg['announcements']
    if not ann:
        return "<div style='padding:20px;text-align:center;color:#999'>暂无公告</div>"
    html = '<div style="max-height:220px;overflow-y:auto">'
    for a in ann:
        html += f'<div style="background:#e8f5e9;padding:9px;border-radius:8px;margin:4px">📢{a}</div>'
    html += "</div>"
    return html


def export_excel(filter_st="全部"):
    rep, _ = load_reports()
    if filter_st != "全部":
        rep = [x for x in rep if x['status'] == filter_st]
    data_list = []
    for r in rep:
        abn_txt = ";".join([f"{o['class_zh']}:{o['abnormal']}" for o in r['objects']])
        data_list.append({
            "单号": r['id'],
            "上报时间": r['timestamp'][:19],
            "位置": r['location'],
            "上报人": r['reporter'],
            "状态": r['status'],
            "异常内容": abn_txt,
            "分派队伍": r.get("assignee", ""),
            "管理员备注": r.get("admin_feedback", ""),
            "驳回原因": r.get("reject_reason", "")
        })
    df = pd.DataFrame(data_list)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="工单明细", index=False)
    buf.seek(0)
    return buf


config_data = load_config()
team_list = config_data['repair_teams']

with gr.Blocks(title="路况随手拍-管理后台", theme=gr.themes.Soft(), css="footer{visibility:hidden}") as demo:
    gr.HTML(
        '<div style="background:linear-gradient(135deg,#2c3e50,#3498db);padding:20px;border-radius:16px;color:white;text-align:center">'
        '<h1>🏛 市政路况管理后台</h1></div>'
    )

    with gr.Tabs():
        with gr.TabItem("📊 数据仪表盘"):
            dash_html = gr.HTML(value=dashboard_html())
            dash_ref = gr.Button("刷新统计")

        with gr.TabItem("🔍 审核中心"):
            gr.Markdown("### 待审核工单列表")
            with gr.Row():
                audit_refresh_btn = gr.Button("🔄 刷新审核列表", size="sm")
            audit_list_html = gr.HTML(value=get_reports_html(filter_status="待审核"))
            with gr.Row():
                audit_select = gr.Dropdown(
                    choices=get_report_choices("待审核"),
                    label="选择待审核工单（选中自动加载图片）",
                    allow_custom_value=True,
                    scale=3
                )
                audit_load_btn = gr.Button("📂 加载工单", scale=1)
            with gr.Row():
                with gr.Column():
                    audit_orig_img = gr.Image(label="📷 市民上报原图", type="filepath", height=250)
                    audit_ann_img = gr.Image(label="🤖 AI识别标注图", type="filepath", height=250)
                with gr.Column():
                    audit_detail = gr.Textbox(label="工单详细信息", lines=14, interactive=False)
                    audit_note = gr.Textbox(label="审核备注/驳回原因", placeholder="通过可不填，驳回必须填写原因", lines=2)
                    with gr.Row():
                        audit_pass_btn = gr.Button("✅ 审核通过", variant="primary")
                        audit_reject_btn = gr.Button("❌ 驳回工单", variant="stop")
            audit_msg = gr.Textbox(label="审核反馈", interactive=False)

        with gr.TabItem("📋 工单管理"):
            with gr.Row():
                s_in = gr.Textbox(label="搜索(单号/地址/姓名)", scale=3)
                st_sel = gr.Dropdown(
                    choices=["全部", "待审核", "审核通过", "处理中", "已整改", "已驳回"],
                    value="全部", label="状态筛选", scale=1
                )
                bt_fil = gr.Button("筛选", scale=1)
                bt_ref = gr.Button("刷新全部", scale=1)
            list_html = gr.HTML(value=get_reports_html())

            gr.Markdown("### 🔧 工单选择与详情")
            with gr.Row():
                quick_select = gr.Dropdown(
                    choices=get_report_choices(),
                    label="选择工单查看详情",
                    allow_custom_value=True,
                    scale=3,
                    info="选中后自动加载图片和详情"
                )
            with gr.Row():
                with gr.Column():
                    mg_detail = gr.Textbox(label="工单详情", lines=16, interactive=False)
                with gr.Column():
                    mg_orig_img = gr.Image(label="📷 上报原图", type="filepath", height=180)
                    mg_ann_img = gr.Image(label="🤖 AI标注图", type="filepath", height=180)
                    mg_repair_img = gr.Image(label="🔧 维修过程图", type="filepath", height=180)
                    mg_after_img = gr.Image(label="✅ 整改后照片", type="filepath", height=180)

            gr.Markdown("### ⚡ 快捷操作")
            with gr.Row():
                quick_detail = gr.Button("📄 查看详情")
                quick_assign = gr.Button("🚀 分派处理")
                quick_finish = gr.Button("✅ 办结归档")
                quick_reject = gr.Button("❌ 驳回工单")
                quick_del = gr.Button("🗑️ 删除工单")

            gr.Markdown("### ✏️ 手动操作")
            with gr.Row():
                op_rid = gr.Textbox(label="工单编号", placeholder="粘贴工单ID")
                op_type = gr.Dropdown(
                    choices=["分派处理", "办结归档", "驳回工单", "删除"],
                    label="操作类型"
                )
                op_team = gr.Dropdown(choices=team_list, label="分派队伍")
                op_note = gr.Textbox(label="备注/驳回说明")
                op_btn = gr.Button("执行操作", variant="primary")
            op_msg = gr.Textbox(label="执行反馈", interactive=False)

        with gr.TabItem("✅ 验收中心"):
            gr.Markdown("### 维修完成待验收工单")
            with gr.Row():
                confirm_refresh_btn = gr.Button("🔄 刷新验收列表", size="sm")
            confirm_list_html = gr.HTML(value=get_reports_html(filter_status="已整改"))
            with gr.Row():
                confirm_select = gr.Dropdown(
                    choices=get_report_choices("已整改"),
                    label="选择已整改工单查看维修结果",
                    allow_custom_value=True,
                    scale=3
                )
            with gr.Row():
                with gr.Column():
                    confirm_detail = gr.Textbox(label="工单详情", lines=14, interactive=False)
                    confirm_note = gr.Textbox(label="验收备注", placeholder="填写验收意见...", lines=2)
                    confirm_btn = gr.Button("✅ 确认验收通过", variant="primary")
                    confirm_msg = gr.Textbox(label="验收反馈", interactive=False)
                with gr.Column():
                    confirm_orig = gr.Image(label="📷 原图对比", type="filepath", height=200)
                    confirm_repair_img = gr.Image(label="🔧 维修过程图", type="filepath", height=200)
                    confirm_after = gr.Image(label="✅ 整改后照片", type="filepath", height=200)

        with gr.TabItem("📥 Excel导出"):
            with gr.Row():
                exp_refresh_btn = gr.Button("🔄 刷新数据", size="sm")
            exp_sel = gr.Dropdown(
                choices=["全部", "待审核", "审核通过", "处理中", "已整改", "已驳回"],
                value="全部", label="导出筛选"
            )
            exp_btn = gr.Button("导出Excel")
            exp_file = gr.File(label="下载文件")

        with gr.TabItem("⚙ 公告管理"):
            with gr.Row():
                ann_refresh_btn = gr.Button("🔄 刷新公告", size="sm")
            ann_in = gr.Textbox(label="新增公告内容")
            ann_add = gr.Button("添加公告")
            ann_msg = gr.Textbox(label="操作提示", interactive=False)
            ann_view = gr.HTML(value=get_ann_html())

    # 事件绑定
    dash_ref.click(dashboard_html, [], [dash_html])

    def refresh_all():
        return (
            get_reports_html(),
            gr.update(choices=get_report_choices()),
            gr.update(choices=get_report_choices("待审核")),
            get_reports_html(filter_status="待审核"),
            get_reports_html(filter_status="已整改"),
            gr.update(choices=get_report_choices("已整改"))
        )

    def do_filter(s, st):
        return (
            get_reports_html(s, st),
            gr.update(choices=get_report_choices()),
            gr.update(choices=get_report_choices("待审核")),
            get_reports_html(filter_status="待审核"),
            get_reports_html(filter_status="已整改"),
            gr.update(choices=get_report_choices("已整改"))
        )

    bt_fil.click(do_filter, [s_in, st_sel], [list_html, quick_select, audit_select, audit_list_html, confirm_list_html, confirm_select])
    bt_ref.click(refresh_all, [], [list_html, quick_select, audit_select, audit_list_html, confirm_list_html, confirm_select])

    def on_audit_select_change(choice):
        rid = parse_choice(choice)
        if not rid:
            return "", None, None
        detail, orig, ann, _, _, _, _, _, _ = get_detail_data(choice)
        return detail, orig, ann

    audit_select.change(on_audit_select_change, [audit_select], [audit_detail, audit_orig_img, audit_ann_img])
    audit_load_btn.click(on_audit_select_change, [audit_select], [audit_detail, audit_orig_img, audit_ann_img])

    def do_audit_pass(choice, note):
        rid = parse_choice(choice)
        if not rid:
            return "❌ 请选择工单", get_reports_html(filter_status="待审核"), None, None, ""
        msg, html, orig, ann, detail = audit_report(rid, '通过', note)
        return msg, html, orig, ann, detail

    def do_audit_reject(choice, note):
        rid = parse_choice(choice)
        if not rid:
            return "❌ 请选择工单", get_reports_html(filter_status="待审核"), None, None, ""
        if not note or not note.strip():
            return "❌ 驳回必须填写原因", get_reports_html(filter_status="待审核"), None, None, ""
        msg, html, orig, ann, detail = audit_report(rid, '驳回', note)
        return msg, html, orig, ann, detail

    audit_pass_btn.click(
        do_audit_pass,
        [audit_select, audit_note],
        [audit_msg, audit_list_html, audit_orig_img, audit_ann_img, audit_detail]
    )
    audit_reject_btn.click(
        do_audit_reject,
        [audit_select, audit_note],
        [audit_msg, audit_list_html, audit_orig_img, audit_ann_img, audit_detail]
    )

    # 审核中心刷新按钮
    def refresh_audit_tab():
        return (
            get_reports_html(filter_status="待审核"),
            gr.update(choices=get_report_choices("待审核")),
            "",
            None,
            None,
            ""
        )
    audit_refresh_btn.click(
        refresh_audit_tab,
        [],
        [audit_list_html, audit_select, audit_msg, audit_orig_img, audit_ann_img, audit_detail]
    )

    def on_mgmt_select_change(choice):
        return get_detail_data(choice)
    quick_select.change(on_mgmt_select_change, [quick_select], [mg_detail, mg_orig_img, mg_ann_img, mg_repair_img, mg_after_img, gr.State(), gr.State(), gr.State(), gr.State()])

    def quick_show_detail(rid_raw):
        rid = parse_choice(rid_raw)
        if not rid:
            return "❌ 请先选择工单", "", None, None, None, None
        detail, orig, ann, rp, af, st, loc, rn, rs = get_detail_data(rid_raw)
        return "", detail, orig, rp, af

    def quick_set_type(rid_raw, t):
        rid = parse_choice(rid_raw)
        if not rid:
            return f"❌ 请先选择工单，再点击{t}", t, rid_raw
        return "", t, rid_raw

    quick_detail.click(quick_show_detail, [quick_select], [op_msg, mg_detail, mg_orig_img, mg_repair_img, mg_after_img])
    quick_assign.click(lambda x: quick_set_type(x, "分派处理"), [quick_select], [op_msg, op_type, op_rid])
    quick_finish.click(lambda x: quick_set_type(x, "办结归档"), [quick_select], [op_msg, op_type, op_rid])
    quick_reject.click(lambda x: quick_set_type(x, "驳回工单"), [quick_select], [op_msg, op_type, op_rid])
    quick_del.click(lambda x: quick_set_type(x, "删除"), [quick_select], [op_msg, op_type, op_rid])

    def run_op(rid_raw, opt, tm, nt):
        rid = parse_choice(rid_raw)
        if not rid:
            return "❌ 请输入工单编号", get_reports_html(), gr.update(), gr.update(), gr.update(), "", None, None, None, None
        if opt == "分派处理":
            msg, html = change_report_status(rid, "处理中", assignee=tm, feedback=nt)
        elif opt == "办结归档":
            msg, html = change_report_status(rid, "已整改", feedback=nt)
        elif opt == "驳回工单":
            msg, html = change_report_status(rid, "已驳回", reject_note=nt)
        elif opt == "删除":
            msg, html = delete_single(rid)
        else:
            msg, html = "❌ 请选择操作类型", get_reports_html()
        detail, orig, ann, rp, af, st, loc, rn, rs = get_detail_data(rid_raw) if opt != "删除" else ("", None, None, None, None, "", "", "", "")
        return msg, html, gr.update(choices=get_report_choices()), gr.update(choices=get_report_choices("待审核")), gr.update(choices=get_report_choices("已整改")), detail, orig, rp, af

    op_btn.click(
        run_op,
        [op_rid, op_type, op_team, op_note],
        [op_msg, list_html, quick_select, audit_select, confirm_select, mg_detail, mg_orig_img, mg_repair_img, mg_after_img]
    )

    # 验收中心
    def on_confirm_select_change(choice):
        return get_detail_data(choice)
    confirm_select.change(
        on_confirm_select_change,
        [confirm_select],
        [confirm_detail, confirm_orig, gr.State(), confirm_repair_img, confirm_after, gr.State(), gr.State(), gr.State(), gr.State()]
    )

    def do_confirm(choice, note):
        rid = parse_choice(choice)
        if not rid:
            return "❌ 请选择工单", "", None, None, None, None, get_reports_html(filter_status="已整改")
        msg, detail, orig, ann, rp, af, html = confirm_repair_fn(rid, note)
        return msg, detail, orig, rp, af, gr.update(choices=get_report_choices("已整改")), html

    confirm_btn.click(
        do_confirm,
        [confirm_select, confirm_note],
        [confirm_msg, confirm_detail, confirm_orig, confirm_repair_img, confirm_after, confirm_select, confirm_list_html]
    )

    # 验收中心刷新按钮
    def refresh_confirm_tab():
        return (
            get_reports_html(filter_status="已整改"),
            gr.update(choices=get_report_choices("已整改")),
            "",
            "",
            None,
            None,
            None
        )
    confirm_refresh_btn.click(
        refresh_confirm_tab,
        [],
        [confirm_list_html, confirm_select, confirm_msg, confirm_detail, confirm_orig, confirm_repair_img, confirm_after]
    )

    exp_btn.click(export_excel, [exp_sel], [exp_file])
    ann_add.click(add_ann, [ann_in], [ann_msg, ann_view])

    # 公告刷新按钮
    def refresh_ann_tab():
        return get_ann_html(), ""
    ann_refresh_btn.click(refresh_ann_tab, [], [ann_view, ann_msg])

    # Excel刷新按钮
    def refresh_exp_tab():
        return gr.update(choices=["全部", "待审核", "审核通过", "处理中", "已整改", "已驳回"]), ""
    exp_refresh_btn.click(refresh_exp_tab, [], [exp_sel, exp_file])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8081, share=False)