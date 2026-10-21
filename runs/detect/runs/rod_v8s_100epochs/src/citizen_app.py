import gradio as gr
from ultralytics import YOLO
from datetime import datetime
import json
import os
import urllib.request
import uuid
from PIL import Image
import numpy as np
import shutil

MODEL_PATH = 'best.pt'
REPORTS_FILE = 'citizen_reports.json'
SHARED_DATA_FILE = 'shared_reports.json'
UPLOAD_DIR = os.path.abspath('uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    'Bike': '自行车', 'Building': '建筑物', 'Car': '汽车', 'Person': '行人',
    'Stairs': '楼梯', 'Traffic sign': '交通标志', 'Electrical Pole': '电线杆',
    'Road': '道路', 'Motorcycle': '摩托车', 'Dustbin': '垃圾桶', 'Dog': '狗',
    'Manhole': '井盖', 'Tree': '树木', 'Guard rail': '护栏',
    'Pedestrian crosswalk': '斑马线', 'Truck': '卡车', 'Bus': '公交车',
    'Bench': '长椅', 'Traffic Cone': '雪糕筒', 'Fire hydrant': '消防栓',
    'Teraffic Barrel': '防撞桶', 'Plant Pot': '花盆', 'Electrical Box': '电箱',
    'Chair': '椅子', 'Bicycle Rack': '自行车架'
}

ABNORMAL_TYPES = {
    '自行车': ['车辆损坏', '违规停放', '占用人行道', '共享单车堆积', '其他'],
    '建筑物': ['外墙脱落', '玻璃破损', '招牌倾斜', '施工无围挡', '其他'],
    '汽车': ['车辆损坏', '违停挡道', '僵尸车', '车窗破损', '其他'],
    '行人': ['行人摔倒', '行为异常', '需要帮助', '聚众堵塞', '其他'],
    '井盖': ['井盖缺失', '井盖破损', '井盖翻开', '周围塌陷', '其他'],
    '护栏': ['护栏损坏', '护栏变形', '护栏缺失', '护栏倾倒', '其他'],
    '消防栓': ['消防栓倾倒', '消防栓漏水', '消防栓被遮挡', '消防栓破损', '其他'],
    '垃圾桶': ['垃圾桶倾倒', '垃圾满溢', '桶体破损', '异味严重', '其他'],
    '树木': ['树木倾倒', '树枝断裂', '树根拱起', '病虫害', '其他'],
    '交通标志': ['标志倾斜', '标志破损', '标志被遮挡', '内容模糊', '其他'],
    '斑马线': ['斑马线模糊', '斑马线损坏', '被车辆占用', '夜间看不清', '其他'],
}
other_cls = ['卡车', '公交车', '摩托车', '长椅', '椅子', '雪糕筒', '防撞桶', '花盆', '电箱', '自行车架', '楼梯', '道路']
for k in other_cls:
    if k not in ABNORMAL_TYPES:
        ABNORMAL_TYPES[k] = ['设施损坏', '位置不当', '存在隐患', '影响通行', '其他']

reports = []
user_points = {}


def safe_path(p):
    """确保返回给 Gradio 的路径是有效的文件路径"""
    if p and isinstance(p, str) and os.path.isfile(p):
        return p
    return None


def load_data():
    global reports, user_points
    if os.path.exists(SHARED_DATA_FILE):
        try:
            with open(SHARED_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                reports = data.get('reports', [])
                user_points = data.get('user_points', {})
        except Exception as e:
            print(f"加载数据失败: {e}")
            reports = []
            user_points = {}
    else:
        reports = []
        user_points = {}
    save_data()


def save_data():
    try:
        payload = {
            'reports': reports,
            'user_points': user_points,
            'last_update': datetime.now().isoformat()
        }
        with open(SHARED_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(REPORTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存数据失败: {e}")


def save_image(img_input, prefix='img'):
    if img_input is None:
        return None
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    fpath = os.path.join(UPLOAD_DIR, fname)
    try:
        # 字符串路径（Gradio File/Gallery 上传返回的路径）
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


def get_icon_for_class(class_zh):
    icons = {
        '自行车': '🚲', '建筑物': '🏢', '汽车': '🚗', '行人': '🚶', '井盖': '🕳️',
        '护栏': '🚧', '消防栓': '🔥', '垃圾桶': '🗑️', '树木': '🌳', '交通标志': '🚏'
    }
    return icons.get(class_zh, '📍')


def get_gps_location():
    try:
        req = urllib.request.Request('http://ip-api.com/json/?lang=zh-CN', timeout=3)
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode())
        if data['status'] == 'success':
            return f"{data['regionName']}{data['city']}{data.get('district', '')}"
    except Exception:
        pass
    return "福建省莆田市城厢区"


def get_report_choices():
    load_data()
    choices = []
    for r in reports:
        status_emoji = {"待审核": "🟡", "审核通过": "🟣", "处理中": "🔵", "已整改": "🟢", "已驳回": "🔴"}.get(r['status'], "⚪")
        objs = ", ".join([f"{o['class_zh']}:{o['abnormal']}" for o in r['objects'][:2]])
        if len(r['objects']) > 2:
            objs += "..."
        label = f"{r['id']} | {status_emoji}{r['status']} | 📍{r['location'][:12]}... | {objs[:20]}"
        choices.append((label, r['id']))
    return choices


def parse_choice(choice):
    if not choice:
        return ""
    if " | " in choice:
        return choice.split(" | ")[0]
    return choice.strip()


def detect_objects(img, conf_threshold=0.25):
    try:
        results = model.predict(img, conf=conf_threshold)
        result = results[0]
        annotated_img = result.plot()
        objects = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            en_name = result.names[class_id]
            zh_name = CLASS_NAMES.get(en_name, en_name)
            objects.append({
                'class_zh': zh_name,
                'confidence': f"{conf:.0%}",
                'abnormal_options': ABNORMAL_TYPES.get(zh_name, ['需要关注', '其他']),
                'selected_abnormal': None
            })
        return annotated_img, objects, f"✅ 识别完成！发现 {len(objects)} 个物体"
    except Exception as e:
        print(f"识别异常:{e}")
        return None, [], f"❌ 识别失败: {str(e)}"


def render_objects_html(objects):
    if not objects:
        return "<div style='text-align:center; padding:20px; color:#999'>暂无识别结果</div>"
    html = '<div style="max-height:400px; overflow-y:auto; border:1px solid #ddd; border-radius:10px; padding:15px">'
    for obj_item in objects:
        item_zh = obj_item["class_zh"]
        item_conf = obj_item["confidence"]
        item_opt_list = obj_item["abnormal_options"]
        opts_text = "、".join(item_opt_list[:3]) + ("..." if len(item_opt_list) > 3 else "")
        html += (
            f'<div style="border:2px solid #28a745; border-radius:10px; margin:10px 0; padding:15px">'
            f'<div style="display:flex; align-items:center; margin-bottom:10px; flex-wrap:wrap">'
            f'<span style="font-size:24px; margin-right:10px">{get_icon_for_class(item_zh)}</span>'
            f'<span style="font-weight:bold; font-size:16px">{item_zh}</span>'
            f'<span style="margin-left:10px; background:#28a745; color:white; padding:2px 10px; border-radius:20px; font-size:12px">置信度: {item_conf}</span>'
            f'</div>'
            f'<div style="font-size:13px; color:#666">可选异常：{opts_text}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def get_abnormal_choices(objects):
    choices = []
    for obj in objects:
        zh = obj['class_zh']
        for opt in obj['abnormal_options']:
            choices.append(f"{zh}:{opt}")
    return choices


def on_detect(img, hide_low_conf):
    if img is None:
        return [None, [], "请先上传图片", "", gr.update(choices=[], value=[])]
    conf_th = 0.5 if hide_low_conf else 0.25
    annotated_img, objects, msg = detect_objects(img, conf_th)
    html = render_objects_html(objects)
    choices = get_abnormal_choices(objects)
    return [annotated_img, objects, msg, html, gr.update(choices=choices, value=[])]


def on_submit(objects_state, selected_abnormals, location, reporter, notes, custom_report, extra_files, input_img, annotated_img):
    try:
        to_report = []
        if selected_abnormals:
            for sel in selected_abnormals:
                if ':' in sel:
                    cls_zh, abn = sel.split(':', 1)
                    to_report.append({'class_zh': cls_zh, 'abnormal': abn})
        if custom_report and custom_report.strip():
            to_report.append({'class_zh': '其他问题', 'abnormal': custom_report.strip()})
        if not to_report:
            return "⚠️ 请至少选择一项异常或填写自定义问题", "", gr.update(), None, None

        orig_path = save_image(input_img, 'orig') if input_img is not None else None
        ann_path = save_image(annotated_img, 'annotated') if annotated_img is not None else None
        extra_paths = []
        if extra_files:
            for f in extra_files:
                try:
                    ext = os.path.splitext(f.name)[1] or '.jpg'
                    fname = f"extra_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}{ext}"
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    shutil.copy(f.name, fpath)
                    extra_paths.append(fpath)
                except Exception as e:
                    print(f"复制补充图片失败: {e}")

        report_id = f"RD{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4]}"
        report_data = {
            'id': report_id,
            'timestamp': datetime.now().isoformat(),
            'location': location.strip() if location else '位置未填',
            'reporter': reporter.strip() if reporter else '匿名',
            'notes': notes.strip() if notes else '',
            'objects': to_report,
            'status': '待审核',
            'original_image': orig_path,
            'annotated_image': ann_path,
            'extra_files': extra_paths,
            'admin_feedback': '',
            'reject_reason': '',
            'assignee': '',
            'audit_info': {},
            'repair': {
                'team': '',
                'accept_time': '',
                'complete_time': '',
                'repair_photos': [],
                'after_photos': [],
                'repair_notes': '',
                'repair_status': ''
            }
        }
        reports.insert(0, report_data)
        rep_name = report_data['reporter']
        if rep_name != '匿名':
            user_points[rep_name] = user_points.get(rep_name, 0) + 10
        save_data()

        report_html = (
            f'<div style="background:#28a745; padding:20px; border-radius:15px; color:white; margin-bottom:15px">'
            f'<h2>✅ 上报成功！</h2><p>单号：{report_id}</p>'
            f'<p style="font-size:13px; opacity:0.9">当前状态：待审核，管理员审核通过后将安排维修</p></div>'
        )
        for item in to_report:
            z_name = item["class_zh"]
            abn_text = item["abnormal"]
            report_html += (
                f'<div style="background:#f8f9fa; padding:10px; margin:8px 0; border-radius:8px">'
                f'{get_icon_for_class(z_name)} {z_name}: ⚠️ {abn_text}</div>'
            )
        if notes:
            report_html += (
                f'<div style="background:#e9ecef; padding:10px; margin:8px 0; border-radius:8px">'
                f'📝 补充说明：{notes}</div>'
            )
        return report_html, f"✅ 上报成功！单号：{report_id}", gr.update(choices=get_report_choices()), None, None
    except Exception as e:
        err = f"❌ 提交失败:{str(e)}"
        return err, err, gr.update(), None, None


def cancel_report(choice):
    rid = parse_choice(choice)
    if not rid:
        return "❌ 请选择工单", get_my_reports_html(), gr.update(choices=get_report_choices())
    target_item = None
    for r in reports:
        if r['id'] == rid:
            target_item = r
            break
    if not target_item:
        return f"❌ 未找到工单{rid}", get_my_reports_html(), gr.update(choices=get_report_choices())
    if target_item.get('status') != '待审核':
        return f"❌ 工单{rid}已进入处理流程，无法撤销", get_my_reports_html(), gr.update(choices=get_report_choices())
    reports.remove(target_item)
    rep_name = target_item.get('reporter', '')
    if rep_name != '匿名' and user_points.get(rep_name, 0) >= 10:
        user_points[rep_name] -= 10
    save_data()
    return f"✅ 撤销成功 {rid}", get_my_reports_html(), gr.update(choices=get_report_choices())


def get_report_detail(choice):
    rid = parse_choice(choice)
    if not rid:
        return "❌ 请选择工单", None, None, None, None
    load_data()
    for r in reports:
        if r['id'] == rid:
            obj_lines = []
            for o in r["objects"]:
                obj_lines.append(f"-{o['class_zh']}:{o['abnormal']}")
            obj_txt = "\n".join(obj_lines)
            detail = (
                f"单号:{r['id']}\n"
                f"时间:{r['timestamp'][:19]}\n"
                f"位置:{r['location']}\n"
                f"上报人:{r['reporter']}\n"
                f"状态:{r['status']}\n"
                f"异常明细：\n{obj_txt}\n"
                f"补充备注：{r.get('notes','无')}\n"
            )
            if r.get('reject_reason'):
                detail += f"驳回原因：{r['reject_reason']}\n"
            if r.get('assignee'):
                detail += f"维修队伍：{r['assignee']}\n"
            repair = r.get('repair', {})
            if repair.get('repair_notes'):
                detail += f"维修备注：{repair['repair_notes']}\n"
            if repair.get('complete_time'):
                detail += f"完成时间：{repair['complete_time'][:19]}"
            orig = safe_path(r.get('original_image'))
            after = repair.get('after_photos', [])
            after_path = safe_path(after[0]) if after else None
            repair_paths = repair.get('repair_photos', [])
            repair_path = safe_path(repair_paths[0]) if repair_paths else None
            return detail, orig, after_path, repair_path, r.get('status', '')
    return "❌ 无此工单", None, None, None, ""


def get_progress_html(choice):
    rid = parse_choice(choice)
    if not rid:
        return "<div style='padding:20px;color:#999'>请选择工单查看进度</div>"
    load_data()
    for r in reports:
        if r['id'] == rid:
            st = r.get('status', '')
            steps = [
                ("1. 市民上报", "待审核", "市民提交问题"),
                ("2. 管理员审核", "审核通过", "核实问题真实性"),
                ("3. 分派维修", "处理中", "分配维修队伍"),
                ("4. 维修整改", "已整改", "现场修复并验收")
            ]
            status_order = {"待审核": 0, "审核通过": 1, "处理中": 2, "已整改": 3, "已驳回": -1}
            curr_idx = status_order.get(st, 0)
            if st == "已驳回":
                return (
                    f'<div style="padding:15px; background:#f8d7da; border-radius:10px; color:#721c24; margin:10px 0">'
                    f'<h4>❌ 该工单已被驳回</h4><p>原因：{r.get("reject_reason","无")}</p></div>'
                )
            html = '<div style="padding:15px"><h4>📍 处理进度</h4>'
            for i, (label, target_st, desc) in enumerate(steps):
                if i < curr_idx:
                    bg = '#28a745'
                    icon = '✅'
                    border = '2px solid #28a745'
                    status_text = '已完成'
                elif i == curr_idx:
                    bg = '#fd7e14'
                    icon = '🔵'
                    border = '3px solid #fd7e14'
                    status_text = '进行中'
                else:
                    bg = '#e9ecef'
                    icon = '⬜'
                    border = '2px solid #dee2e6'
                    status_text = '等待中'
                html += (
                    f'<div style="display:flex; align-items:center; margin:12px 0; padding:12px; border-radius:10px; background:{"#f8fff8" if i <= curr_idx else "#f8f9fa"}; border:{border}">'
                    f'<div style="min-width:45px; height:45px; border-radius:50%; background:{bg}; color:white; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:bold; margin-right:15px">{icon}</div>'
                    f'<div style="flex:1">'
                    f'<div style="font-weight:bold; font-size:15px; color:#333">{label}</div>'
                    f'<div style="font-size:12px; color:#666; margin-top:2px">{desc}</div>'
                    f'</div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-size:13px; color:{bg if i <= curr_idx else "#999"}; font-weight:bold">{status_text}</div>'
                    f'<div style="font-size:11px; color:#888">{target_st}</div>'
                    f'</div>'
                    f'</div>'
                )
            html += '</div>'
            return html
    return "<div style='padding:20px;color:#999'>未找到工单</div>"


def get_my_reports_html():
    load_data()
    if not reports:
        return '<div style="text-align:center; padding:50px; color:#999">暂无上报记录</div>'
    html = '<div style="max-height:500px; overflow-y:auto">'
    for r in reports[:30]:
        status = r.get('status', '')
        color_map = {
            '待审核': '#ffc107', '审核通过': '#6f42c1', '处理中': '#17a2b8',
            '已整改': '#28a745', '已驳回': '#dc3545'
        }
        color = color_map.get(status, '#888')
        obj_html = ""
        for o in r["objects"][:3]:
            z_name = o["class_zh"]
            abn = o["abnormal"]
            obj_html += (
                f'<span style="background:#e9ecef; padding:2px 8px; border-radius:15px; font-size:12px; margin:2px">'
                f'{z_name}:{abn}</span>'
            )
        repair = r.get('repair', {})
        after_photos = repair.get('after_photos', [])
        has_after = len(after_photos) > 0
        after_badge = '<span style="background:#28a745; color:white; padding:2px 8px; border-radius:10px; font-size:11px; margin-left:5px">📷 已上传整改图</span>' if has_after else ''
        img_badge = '<span style="background:#007bff; color:white; padding:2px 6px; border-radius:8px; font-size:11px; margin-left:5px">📷</span>' if r.get('original_image') else ''
        html += (
            f'<div style="border:1px solid #ddd; border-radius:12px; margin:12px; padding:15px; background:white">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap">'
            f'<div><b>📋 {r["id"]}</b> <span style="background:{color}; color:white; padding:2px 10px; border-radius:20px; font-size:12px">{status}</span>{after_badge}{img_badge}</div>'
            f'<div style="font-size:12px; color:#666">📅 {r["timestamp"][:19]}</div>'
            f'</div>'
            f'<div style="margin-top:8px; font-size:13px">📍 {r.get("location","未知")} | 👤 {r.get("reporter","匿名")}</div>'
            f'<div style="margin-top:5px">{obj_html}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def add_tag_1(notes): return notes + "；积水严重" if notes else "积水严重"
def add_tag_2(notes): return notes + "；夜间无照明" if notes else "夜间无照明"
def add_tag_3(notes): return notes + "；占道影响通行" if notes else "占道影响通行"
def add_tag_4(notes): return notes + "；存在安全隐患" if notes else "存在安全隐患"


def on_select_change(choice):
    rid = parse_choice(choice)
    if not rid:
        return "", None, None, None, None, "<div style='padding:20px;color:#999'>请选择工单</div>"
    detail, orig, after, repair_img, status = get_report_detail(choice)
    progress = get_progress_html(choice)
    return detail, orig, after, repair_img, status, progress


def refresh_my_reports():
    load_data()
    return [
        get_my_reports_html(),
        gr.update(choices=get_report_choices()),
        "",
        "",
        None,
        None,
        None,
        "<div style='padding:20px;color:#999'>请选择工单</div>"
    ]


# ============ 视障辅助检测模块 ============
def detect_for_accessibility(img, conf_threshold=0.3, user_type="视障人士"):
    """面向视障人士和低头族的实时障碍物检测与语音播报"""
    try:
        if img is None:
            return None, "请先上传或拍摄图片"
        results = model.predict(img, conf=conf_threshold)
        result = results[0]
        annotated_img = result.plot()

        # 危险等级分类
        high_risk = ['汽车', '摩托车', '卡车', '公交车', '行人', '狗', '井盖', '楼梯']
        medium_risk = ['自行车', '垃圾桶', '消防栓', '护栏', '电线杆', '交通标志', '雪糕筒', '防撞桶']
        low_risk = ['树木', '建筑物', '长椅', '椅子', '花盆', '电箱', '自行车架', '斑马线', '道路']

        detected_objects = []
        warnings = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            en_name = result.names[class_id]
            zh_name = CLASS_NAMES.get(en_name, en_name)

            # 计算距离估算（基于框的大小）
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            img_area = result.orig_shape[0] * result.orig_shape[1]
            ratio = box_area / img_area

            if ratio > 0.15:
                distance = "很近"
            elif ratio > 0.05:
                distance = "较近"
            elif ratio > 0.01:
                distance = "前方"
            else:
                distance = "远处"

            risk_level = "高" if zh_name in high_risk else ("中" if zh_name in medium_risk else "低")

            detected_objects.append({
                'class_zh': zh_name,
                'confidence': f"{conf:.0%}",
                'distance': distance,
                'risk': risk_level,
                'position': '左侧' if (x1 + x2) / 2 < result.orig_shape[1] * 0.4 else ('右侧' if (x1 + x2) / 2 > result.orig_shape[1] * 0.6 else '正前方')
            })

            if risk_level in ["高", "中"]:
                warnings.append(f"{distance}发现{zh_name}，位于{detected_objects[-1]['position']}")

        # 生成语音播报文本
        if not detected_objects:
            voice_text = "前方道路通畅，暂未检测到障碍物。"
        else:
            high_count = sum(1 for o in detected_objects if o['risk'] == '高')
            if high_count > 0:
                voice_text = f"⚠️ 警告！检测到{len(detected_objects)}个障碍物，其中{high_count}个高风险。"
            else:
                voice_text = f"检测到{len(detected_objects)}个障碍物，请注意避让。"

            for w in warnings[:5]:
                voice_text += f" {w}。"

        # 生成详细报告
        detail_html = '<div style="padding:15px">'
        detail_html += f'<h3>🔍 检测结果（{user_type}模式）</h3>'
        detail_html += f'<p>检测到 <b>{len(detected_objects)}</b> 个障碍物</p>'

        for obj in detected_objects:
            risk_color = '#dc3545' if obj['risk'] == '高' else ('#fd7e14' if obj['risk'] == '中' else '#28a745')
            detail_html += (
                f'<div style="border-left:4px solid {risk_color}; padding:10px; margin:8px 0; background:#f8f9fa; border-radius:8px">'
                f'<b>{get_icon_for_class(obj["class_zh"])} {obj["class_zh"]}</b> '
                f'<span style="background:{risk_color}; color:white; padding:2px 8px; border-radius:10px; font-size:12px">{obj["risk"]}风险</span> '
                f'<span style="color:#666; font-size:13px">| 置信度 {obj["confidence"]} | {obj["distance"]} | {obj["position"]}</span>'
                f'</div>'
            )

        detail_html += '</div>'

        return annotated_img, voice_text, detail_html
    except Exception as e:
        print(f"检测异常:{e}")
        return None, f"❌ 检测失败: {str(e)}", ""


def generate_tts(text):
    """生成简单的文本提示（实际可接入TTS引擎）"""
    return f"🗣️ 语音播报内容：\n{text}\n\n💡 提示：实际部署时可接入系统TTS或百度/讯飞语音合成API实现真实语音播报。"


with gr.Blocks(title="路况随手拍", theme=gr.themes.Soft(), css="footer{visibility:hidden}") as demo:
    gr.HTML(
        '<div style="background:#28a745; padding:20px; border-radius:15px; color:white; text-align:center; margin-bottom:20px">'
        '<h1>📸 路况随手拍</h1><p>拍一拍 · 识一识 · 共建美好城市</p></div>'
    )

    objects_state = gr.State([])

    with gr.Tabs():
        with gr.TabItem("📸 随手拍上报"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="上传照片", type="numpy", height=280)
                    extra_imgs = gr.File(label="补充图片", file_count="multiple")
                    upload_btn = gr.Button("🔍 开始识别", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="AI标注结果", type="numpy", height=280)
            conf_filter = gr.Checkbox(label="隐藏低置信度目标(<50%)", value=False)
            detect_status = gr.Textbox(label="识别状态", interactive=False)
            abnormal_container = gr.HTML("")
            abnormal_selector = gr.Dropdown(
                label="选择检测到的异常（可多选）",
                multiselect=True,
                choices=[],
                interactive=True,
                info="识别完成后，请在此勾选需要上报的异常类型"
            )
            gr.Markdown("📝 快捷标签")
            with gr.Row():
                tag1 = gr.Button("积水严重", size="sm")
                tag2 = gr.Button("夜间无照明", size="sm")
                tag3 = gr.Button("占道影响通行", size="sm")
                tag4 = gr.Button("存在安全隐患", size="sm")
            with gr.Row():
                with gr.Column(scale=2):
                    location_input = gr.Textbox(label="位置描述", placeholder="例如：莆田市城厢区荔城中大道", lines=2)
                    gps_btn = gr.Button("📍 获取GPS定位", size="sm")
                    reporter_input = gr.Textbox(label="上报人", placeholder="昵称（用于积分榜）")
                    notes_input = gr.Textbox(label="问题描述", placeholder="详细描述...", lines=2)
                    custom_input = gr.Textbox(label="其他问题", placeholder="不在25类的问题", lines=1)
                with gr.Column(scale=1):
                    submit_btn = gr.Button("📤 提交上报", variant="primary")
                    report_result = gr.HTML("")

        with gr.TabItem("📋 我的上报"):
            my_reports = gr.HTML(value=get_my_reports_html())
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                report_id_select = gr.Dropdown(
                    choices=get_report_choices(),
                    label="选择工单查看详情",
                    allow_custom_value=True,
                    scale=3,
                    info="选中工单后自动显示详情、图片和进度"
                )
                cancel_btn = gr.Button("🗑️ 撤销上报", size="sm", variant="stop")
            cancel_status = gr.Textbox(label="撤销反馈", interactive=False)

            with gr.Row():
                with gr.Column():
                    detail_msg = gr.Textbox(label="工单详情", interactive=False, lines=14)
                with gr.Column():
                    detail_orig_img = gr.Image(label="📷 上报原图", type="filepath", height=180)
                    detail_repair_img = gr.Image(label="🔧 维修过程图", type="filepath", height=180)
                    detail_after_img = gr.Image(label="✅ 整改后照片", type="filepath", height=180)
            progress_html = gr.HTML("")

        # 新增：视障辅助检测模块（替换原积分榜）
        with gr.TabItem("👁️ 视障辅助检测"):
            gr.HTML(
                '<div style="background:linear-gradient(135deg,#6f42c1,#8e44ad);padding:15px;border-radius:12px;color:white;text-align:center;margin-bottom:15px">'
                '<h2>👁️ 城市道路障碍物实时检测</h2>'
                '<p>面向视障人士与低头族的安全辅助 · YOLOv8实时感知</p></div>'
            )

            with gr.Row():
                with gr.Column():
                    acc_input_img = gr.Image(label="📷 拍摄或上传前方道路照片", type="numpy", height=300)
                    with gr.Row():
                        acc_user_type = gr.Radio(
                            choices=["视障人士", "低头族/分心行人"],
                            value="视障人士",
                            label="用户类型"
                        )
                        acc_conf = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="检测置信度阈值")
                    acc_detect_btn = gr.Button("🔍 开始安全检测", variant="primary")
                with gr.Column():
                    acc_output_img = gr.Image(label="🤖 AI检测标注结果", type="numpy", height=300)

            acc_voice_text = gr.Textbox(label="🗣️ 语音播报文本", lines=4, interactive=False)
            acc_tts_btn = gr.Button("🔊 生成TTS语音播报提示")
            acc_tts_output = gr.Textbox(label="TTS输出", lines=6, interactive=False)
            acc_detail_html = gr.HTML("")

            gr.Markdown("""
### 📖 使用说明
1. **拍摄前方道路**：使用手机后置摄像头拍摄前方路况
2. **选择用户类型**：
   - **视障人士模式**：重点检测高风险障碍物（车辆、行人、井盖、楼梯等），播报更详细的位置和距离信息
   - **低头族模式**：侧重提醒前方突发障碍物和交通标志
3. **查看检测结果**：AI实时标注所有障碍物，按风险等级分类显示
4. **语音播报**：生成自然语言播报文本，可接入TTS引擎实现真实语音输出

### ⚠️ 安全提示
- 本系统为辅助工具，不能完全替代人工判断
- 过马路时请务必听从交通信号灯和现场引导
- 建议在熟悉的路段使用，复杂环境请寻求他人帮助
            """)

        with gr.TabItem("❓ 帮助"):
            gr.Markdown("""
### 📖 使用说明
1. 上传照片 → 点击「开始识别」
2. 在下方多选框勾选需要上报的异常类型
3. 快捷标签一键填充描述
4. GPS自动填充地址
5. 填写昵称提交

### 处理流程
待审核 → 审核通过 → 处理中 → 已整改

### 视障辅助功能
在「视障辅助检测」标签页，可实时检测前方障碍物，获得语音播报提示，保障出行安全。
            """)

    upload_btn.click(
        on_detect,
        [input_img, conf_filter],
        [output_img, objects_state, detect_status, abnormal_container, abnormal_selector]
    )
    gps_btn.click(get_gps_location, [], [location_input])
    submit_btn.click(
        on_submit,
        [objects_state, abnormal_selector, location_input, reporter_input, notes_input, custom_input, extra_imgs, input_img, output_img],
        [report_result, detect_status, report_id_select, input_img, output_img]
    )
    tag1.click(add_tag_1, [notes_input], [notes_input])
    tag2.click(add_tag_2, [notes_input], [notes_input])
    tag3.click(add_tag_3, [notes_input], [notes_input])
    tag4.click(add_tag_4, [notes_input], [notes_input])

    report_id_select.change(
        on_select_change,
        [report_id_select],
        [detail_msg, detail_orig_img, detail_after_img, detail_repair_img, gr.State(), progress_html]
    )

    refresh_btn.click(
        refresh_my_reports,
        [],
        [my_reports, report_id_select, cancel_status, detail_msg, detail_orig_img, detail_after_img, detail_repair_img, progress_html]
    )
    cancel_btn.click(
        cancel_report,
        [report_id_select],
        [cancel_status, my_reports, report_id_select]
    )

    # 视障辅助检测事件绑定
    acc_detect_btn.click(
        detect_for_accessibility,
        [acc_input_img, acc_conf, acc_user_type],
        [acc_output_img, acc_voice_text, acc_detail_html]
    )
    acc_tts_btn.click(
        generate_tts,
        [acc_voice_text],
        [acc_tts_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=8080)