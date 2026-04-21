
#---------------------------------------
      
    #    uncomment this when needed


#---------------------------------------




# from fastapi import APIRouter, Request
# import psutil
# import platform

# router = APIRouter()


# @router.get("/", summary="Health check")
# async def health(request: Request):
#     registry = request.app.state.model_registry
#     return {
#         "status": "ok",
#         "loaded_models": list(registry.loaded_models.keys()),
#         "system": {
#             "platform": platform.system(),
#             "python": platform.python_version(),
#             "cpu_percent": psutil.cpu_percent(),
#             "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
#             "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
#         },
#     }


# @router.get("/gpu", summary="GPU memory status (Paddle)")
# async def gpu_health():
#     try:
#         import paddle

#         is_cuda = getattr(paddle.device, "is_compiled_with_cuda", lambda: False)()
#         if is_cuda:
#             n = paddle.device.cuda.device_count()
#             if n and int(n) > 0:
#                 return {
#                     "gpu_available": True,
#                     "backend": "paddle",
#                     "device_count": int(n),
#                 }
#     except Exception:
#         pass
#     return {"gpu_available": False, "note": "Paddle CUDA not available or Paddle not installed"}
