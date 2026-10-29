# R2T2 批处理 WebSocket 服务

本 fork 在上游 Confucius4-R2T2 上增加跨会话动态合批。服务入口为
`ws_batch_server.py`，复用 `ws_server.py` 的模型加载、参数、鉴权和文本规则。
模型只在一个独立工作进程中加载，各会话保持独立状态。

## 启动

先按上游 README 安装依赖并准备 ASR、FireRedVAD 模型。在 Linux/CUDA 环境中运行：

```bash
export R2T2_BATCH_ENABLED=1
export R2T2_CHUNK_MS=480
export R2T2_LOOKAHEAD_MS=160
export R2T2_BATCH_MAX_SIZE=8
export R2T2_BATCH_WAIT_MS=5
export R2T2_BATCH_QUEUE_SECONDS=60
python -u ws_batch_server.py \
  --asr_model_path /path/to/Confucius4-R2T2 \
  --vad_model_path /path/to/Stream-VAD \
  --port 8272
```

`run_start_server.sh` 仍启动原服务；批处理服务使用上面的入口。
客户端使用 `/asr_stream_api_v1`，保持原有握手、PCM16 和 EOS 协议。
鉴权沿用 `ws_server.py` 的 `secret_key_list`，部署时设置实际使用的密钥。
`GET /health` 可查看模型进程状态、执行模式、步长和批配置。

## 配置

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `R2T2_BATCH_ENABLED` | `1` | `1`/`true` 启用合批，`0`/`false` 使用原同步串行推理 |
| `R2T2_CHUNK_MS` | `160` | 推理步长，80～2000ms，80 的整数倍 |
| `R2T2_LOOKAHEAD_MS` | `160` | 首轮额外音频，0～2000ms，80 的整数倍 |
| `R2T2_BATCH_MAX_SIZE` | `8` | 最大批大小，1～64 |
| `R2T2_BATCH_WAIT_MS` | `5` | 最大合批等待，0～100ms |
| `R2T2_BATCH_QUEUE_SECONDS` | `30` | 单连接未处理音频上限，1～300s |

配置在启动时读取，修改后需重启服务。480ms 步长加 160ms 前瞻时，首轮需要
640ms 音频。服务最多接纳 16 个活动会话；输入积压超限时明确断开连接。
批处理入口仅提供 v1 路由，原服务仍保留 v0/v1。

返回指标包括批大小、批次推理时间、排队时间及已处理音频进度。
合批模式中的推理时间属于整个共享批次，不能跨会话相加作为 GPU 忙时。
单次合批成功不代表多路输入可以持续实时，需使用目标硬件和音频压测。

## 验证

CPU 回归无需加载模型，需安装 numpy：

```bash
python -m unittest discover -s tests -p test_batching.py -v
```

真实模型冒烟需安装 websockets 并启动服务：

```bash
export R2T2_SECRET_KEY=your-configured-key
python tests/smoke_batch_ws.py /path/to/audio.wav \
  --url ws://localhost:8272/asr_stream_api_v1 \
  --streams 8 --seconds 20 --realtime --output smoke_8.json
```

冒烟报告记录文字哈希和性能指标。真实识别质量仍需使用标注音频评估。
