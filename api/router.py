from fastapi import APIRouter, status
from .model import *
from utils.tools import base64_to_pil, pil_to_base64
from SD_XL.module import bgChanging, inpaint, rmbg

router = APIRouter()


# Health check
@router.get("/api/phototools/health-check")
async def health_check():
    return {"status": "Phototools is running"}


# Inpainting
@router.post("/api/phototools/inpainting", 
                response_model=InpaintingInResponse,
                status_code=status.HTTP_200_OK)
async def inpainting(inpainting_in_request: InpaintingInRequest) -> InpaintingInResponse:
    try:
        image = base64_to_pil(inpainting_in_request.image_base64)
        mask = base64_to_pil(inpainting_in_request.mask_base64)
    except:
        return InpaintingInResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    message="Cannot convert base64 to image",
                                    image_base64="")
    
    prompt = inpainting_in_request.prompt
    negative_prompt = inpainting_in_request.negative_prompt
    
    # Inpainting process
    image = inpaint(image, mask, prompt, negative_prompt)
    
    img_base64 = pil_to_base64(image)
    
    return InpaintingInResponse(status_code=status.HTTP_200_OK, 
                                message="Success",
                                image_base64=img_base64)
    

# Background changing
@router.post("/api/phototools/bg-changing",
                response_model=BgChangingInResponse,
                status_code=status.HTTP_200_OK)
async def bg_changing(bg_changing_in_request: BgChangingInRequest) -> BgChangingInResponse:
    try:
        image = base64_to_pil(bg_changing_in_request.image_base64)
    except:
        return BgChangingInResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    message="Cannot convert base64 to image",
                                    image_base64="")
    
    prompt = bg_changing_in_request.prompt
    negative_prompt = bg_changing_in_request.negative_prompt
    
    # Background changing process
    image = bgChanging(image, prompt, negative_prompt)
    
    img_base64 = pil_to_base64(image)
    
    return BgChangingInResponse(status_code=status.HTTP_200_OK, 
                                message="Success",
                                image_base64=img_base64)
    

# Remove background
@router.post("/api/phototools/remove-bg",
                response_model=BgChangingInResponse,
                status_code=status.HTTP_200_OK)
async def remove_bg(remove_bg_in_request: RemoveBgInRequest) -> RemoveBgInResponse:
    try:
        image = base64_to_pil(remove_bg_in_request.image_base64)
    except:
        return RemoveBgInResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    message="Cannot convert base64 to image",
                                    image_base64="")
    
    # Remove background process
    image = rmbg(image)
    
    img_base64 = pil_to_base64(image)
    
    return RemoveBgInResponse(status_code=status.HTTP_200_OK, 
                                message="Success",
                                image_base64=img_base64)
    

# # Model generation
# @router.post("/api/phototools/model-gen",
#                 response_model=ModelGenInResponse,
#                 status_code=status.HTTP_200_OK)
# async def model_gen(model_gen_in_request: ModelGenInRequest) -> ModelGenInResponse:
#     try:
#         image = Image.open(model_gen_in_request.input_path)
#     except:
#         return ModelGenInResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#                                     message="Cannot read image",
#                                     image_base64="")
    
#     prompt = model_gen_in_request.prompt
#     negative_prompt = model_gen_in_request.negative_prompt
    
#     # Model generation process
#     # ...
#     # ...
    
#     img_base64 = pil_to_base64(image)
    
#     return ModelGenInResponse(status_code=status.HTTP_200_OK, 
#                                 message="Success",
#                                 image_base64=img_base64)
