from __future__ import annotations

from backend.app.models.asset_context import PHOTO_CONTEXT_V2


# Keep the worker idempotency version stable during the gradual v2 rollout.
# New context records carry `schema_version=photo_context_v2`; a later explicit
# reindex can make that schema version a worker-level invalidation boundary.
PHOTO_CONTEXT_VERSION = "photo_context"


PHOTO_CONTEXT_SCHEMA: dict[str, object] = {
    "schema_version": PHOTO_CONTEXT_V2,
    "observations": {
        "caption": "",
        "short_caption": "",
        "objects": [],
        "actions": [],
        "scenes": [],
        "colors": [],
        "visual_attributes": [],
        "visible_people_count": None,
        "visible_text": [],
        "logos": [],
        "relationships": [],
        "notable_regions": [],
    },
    "inferences": {
        "possible_events": [{"value": "", "confidence": 0.0}],
        "possible_locations": [{"value": "", "confidence": 0.0}],
        "possible_time_periods": [{"value": "", "confidence": 0.0}],
        "possible_relationships": [{"value": "", "confidence": 0.0}],
    },
    "raw_ocr_text": "",
    "normalized_ocr_text": "",
}


PHOTO_CONTEXT_PROMPT = (
    "你是本地私人相册的离线视觉上下文采集器。只返回有效 JSON，不要 Markdown。"
    "不要推断人脸真实身份、真实地点、拍摄日期、品牌、人物关系或事件；不确定内容只能写入 inferences 并给出 0 到 1 的 confidence。"
    "observations 只能包含画面中直接可见的事实；metadata 不由你填写；可见文字尽量逐字保留。"
    "字段必须完整："
    "{"
    f"\"schema_version\":\"{PHOTO_CONTEXT_V2}\","
    "\"observations\":{"
    "\"caption\":\"信息密度高的中文可见事实描述\","
    "\"short_caption\":\"一句中文短描述\","
    "\"objects\":[\"直接可见物体\"],\"actions\":[\"直接可见动作\"],\"scenes\":[\"直接可见场景\"],"
    "\"colors\":[\"主要颜色\"],\"visual_attributes\":[\"构图或材质\"],"
    "\"visible_people_count\":null,\"visible_text\":[\"逐字可见文字\"],\"logos\":[\"可确认标识\"],"
    "\"relationships\":[\"可见相对位置或姿态，不含身份\"],\"notable_regions\":[\"局部区域及其可见内容\"]},"
    "\"inferences\":{"
    "\"possible_events\":[{\"value\":\"可能的事件\",\"confidence\":0.0}],"
    "\"possible_locations\":[{\"value\":\"可能的地点类型\",\"confidence\":0.0}],"
    "\"possible_time_periods\":[{\"value\":\"可能的时段或季节\",\"confidence\":0.0}],"
    "\"possible_relationships\":[{\"value\":\"可能的关系\",\"confidence\":0.0}]},"
    "\"raw_ocr_text\":\"可见文字，无法确认则空字符串\","
    "\"normalized_ocr_text\":\"规范化后的可见文字，无法确认则空字符串\""
    "}"
)
