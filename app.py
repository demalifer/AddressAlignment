# -*- coding: utf-8 -*-
"""
FastAPI 服务入口。

这个文件负责把地址对齐能力包装成一个 Web 服务：
1. 服务启动时加载地址标注模型和微调参数。
2. 提供首页 `/`，返回 templates/index.html 前端页面。
3. 提供 POST `/address_alignment`，接收地址文本并返回结构化地址结果。
"""

import config
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from address_alignment import address_alignment
from models_def import AddressTagging, load_params

# 模型在模块加载时初始化一次，避免每个请求都重新加载 BERT 和权重。
# 这里使用 config 中配置的本地预训练模型目录，标签集合来自 config.LABELS。
model = AddressTagging(config.ADDRESS_TAGGING_PRETRAINED_DIR, config.LABELS)

# 加载微调后的地址标注模型参数；如果文件不存在，load_params 会打印提示并保留默认参数。
load_params(model, config.ADDRESS_TAGGING_PARAMS_PATH)

# 创建 FastAPI 应用实例。
app = FastAPI()

# 将 templates 目录挂载为静态资源目录。
# 前端页面中如果引用 /static/... 路径，会从 templates 目录下读取文件。
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "templates"), name="static")


class AddressAlignmentRequest(BaseModel):
    """POST /address_alignment 的请求体。"""

    # message 是用户输入的原始地址文本。
    message: str = Field(..., example="地址文本信息")


class AddressAlignmentResponse(BaseModel):
    """POST /address_alignment 的响应体。"""

    # 这些字段允许为 None，因为模型可能没有识别出某一层级，
    # 或 check_address 在数据库校验时清空了不可信的层级。
    province: str | None = Field(..., example="省份")
    city: str | None = Field(..., example="城市")
    district: str | None = Field(..., example="区县")
    town: str | None = Field(..., example="乡/镇/街道")
    detail: str | None = Field(..., example="详细地址")


@app.get("/")
async def homepage():
    """返回项目自带的前端页面。"""

    return FileResponse(config.BASE_DIR / "templates" / "index.html")


@app.post("/address_alignment")
async def handle_message(request: AddressAlignmentRequest) -> AddressAlignmentResponse:
    """接收一条地址文本，调用 address_alignment 并转换成接口响应格式。"""

    user_message = request.message

    # address_alignment 返回中文 key；这里转换成前端/API 更常用的英文 key。
    address = address_alignment(user_message, model)
    res = AddressAlignmentResponse(
        province=address["省份"],
        city=address["城市"],
        district=address["区县"],
        town=address["街道"],
        detail=address["详细地址"],
    )
    return res


if __name__ == "__main__":
    # 直接运行 app.py 时启动本地服务：
    # 浏览器访问 http://localhost:8089 可以打开首页。
    uvicorn.run(app, host="0.0.0.0", port=8089)
