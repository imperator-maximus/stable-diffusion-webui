import os
import threading

from modules.paths import script_path

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import signal

from ldm.util import instantiate_from_config

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
from modules.ui import plaintext_to_html
import modules.scripts
import modules.processing as processing
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.images as images
import modules.lowvram
import modules.txt2img
import modules.img2img

import time
from io import BytesIO
import json
import base64
from pydantic import BaseModel
from typing import Union
from modules.sd_samplers import samplers, samplers_for_img2img,samplers_k_diffusion

class apiImage(BaseModel):
    # these are base 64 encoded image data
    image: str
    mask: str

    def __getitem__(self, item):
        return getattr(self, item)

class apiInput(BaseModel):
    prompt: str
    neg_prompt: str = ''
    mode: str = 'Not used anymore'
    steps: int = 30
    sampler: str = 'LMS'
    mask_blur: float
    inpainting_fill: Union[str, None] = None
    use_gfpgan: bool = False
    batch_count: int = 1
    cfg_scale: float = 7.0
    denoising_strength: float = 1.0
    seed: int = -1
    height: int = 512
    width: int = 512
    resize_mode: int = 0 # not sure what this one is
    upscaler: str = ''
    upscale_overlap: int = 64
    inpaint_full_res: bool = True
    inpainting_mask_invert: int = 0 # should be bool

    def __getitem__(self, item):
        return getattr(self, item)

class apiInputPlusImage(apiInput):
    initimage: apiImage

modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
realesrgan.setup_realesrgan()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

cached_images = {}


def run_extras(image, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
    processing.torch_gc()

    image = image.convert("RGB")

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    if gfpgan_visibility > 0:
        restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(image, res, gfpgan_visibility)

        image = res

    if codeformer_visibility > 0:
        restored_img = modules.codeformer_model.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(image, res, codeformer_visibility)

        image = res

    if upscaling_resize != 1.0:
        def upscale(image, scaler_index, resize):
            small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
            pixels = tuple(np.array(small).flatten().tolist())
            key = (resize, scaler_index, image.width, image.height, gfpgan_visibility, codeformer_visibility, codeformer_weight) + pixels

            c = cached_images.get(key)
            if c is None:
                upscaler = shared.sd_upscalers[scaler_index]
                c = upscaler.upscale(image, image.width * resize, image.height * resize)
                cached_images[key] = c

            return c

        res = upscale(image, extras_upscaler_1, upscaling_resize)

        if extras_upscaler_2 != 0 and extras_upscaler_2_visibility>0:
            res2 = upscale(image, extras_upscaler_2, upscaling_resize)
            res = Image.blend(res, res2, extras_upscaler_2_visibility)

        image = res

    while len(cached_images) > 2:
        del cached_images[next(iter(cached_images.keys()))]

    images.save_image(image, outpath, "", None, '', opts.samples_format, short_filename=True, no_prompt=True)

    return image, '', ''


def run_pnginfo(image):
    info = ''
    for key, text in image.info.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', '', info


queue_lock = threading.Lock()


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)

modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except Exception:
    pass

sd_config = OmegaConf.load(cmd_opts.config)
shared.sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)
shared.sd_model = (shared.sd_model if cmd_opts.no_half else shared.sd_model.half())

if cmd_opts.lowvram or cmd_opts.medvram:
    modules.lowvram.setup_for_low_vram(shared.sd_model, cmd_opts.medvram)
else:
    shared.sd_model = shared.sd_model.to(shared.device)

modules.sd_hijack.model_hijack.hijack(shared.sd_model)


def webui():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    demo = modules.ui.create_ui(
        txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
        img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
        run_extras=wrap_gradio_gpu_call(run_extras),
        run_pnginfo=run_pnginfo
    )

    (app, local, gradio_remote) = demo.launch(share=False, server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    smp_index=0
    for i in range(0,len(samplers_k_diffusion)):
        if samplers_k_diffusion[i][0]=="LMS": smp_index=i    

    @app.get("/v1/kapi/test")
    async def processTest():
      data={'prompt': 'test', 
            'mode': 'txt2img', 
            'initimage': {'image': '', 'mask': ''}, 
            'steps': 30, 
            'sampler': 'LMS', 
            'mask_blur': 4, 
            'inpainting_fill': 'latent noise', 
            'use_gfpgan': False, 
            'batch_count': 1, 
            'cfg_scale': 5.0, 
            'denoising_strength': 1.0, 
            'seed': -1, 
            'height': 512, 
            'width': 512, 
            'resize_mode': 0, 
            'upscaler': 'RealESRGAN', 
            'upscale_overlap': 64, 
            'inpaint_full_res': True, 
            'inpainting_mask_invert': 0
      } 
      oimages, oinfo, ohtml = modules.txt2img.txt2img(data['prompt'],'',data['steps'],2,data['use_gfpgan'],False,data['batch_count'],1,data['cfg_scale'],data['seed'],data['height'],data['width'],0)
      b64images = []
      for img in oimages:
          buffered = BytesIO()
          img.save(buffered, format="PNG")
          img_str = base64.b64encode(buffered.getvalue())
          b64images.append(img_str.decode())
      return {'images':b64images,'info':oinfo}

    @app.post("/v1/kapi/img2img")
    async def processImg2Img(data: apiInputPlusImage):
        switch_mode = 0
        buffer = BytesIO(base64.b64decode(data['initimage']['image']))
        initimg = Image.open(buffer)
        oimages, oinfo, ohtml = modules.img2img.img2img(data['prompt'],\
                                initimg,\
                                {'image':'', 'mask':''},\
                                data['steps'],\
                                smp_index,\
                                data['mask_blur'],\
                                data['inpainting_fill'],\
                                data['use_gfpgan'],\
                                False,\
                                switch_mode, \
                                data['batch_count'], \
                                1, \
                                data['cfg_scale'],\
                                data['denoising_strength'],\
                                data['seed'],\
                                data['height'],\
                                data['width'],\
                                data['resize_mode'],\
                                data['upscaler'],\
                                data['upscale_overlap'],\
                                data['inpaint_full_res'],\
                                data['inpainting_mask_invert'],\
                                0)
        #
        b64images = []
        for img in oimages:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            b64images.append(img_str.decode())

        return {'images':b64images,'info':oinfo}
 
    @app.post("/v1/kapi/txt2img")
    async def processTxt2Img(data: apiInput):
        oimages, oinfo, ohtml = modules.txt2img.txt2img(data['prompt'],'',data['steps'],smp_index,data['use_gfpgan'],False,data['batch_count'],1,data['cfg_scale'],data['seed'],data['height'],data['width'],0)

        b64images = []
        for img in oimages:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            b64images.append(img_str.decode())

        return {'images':b64images,'info':oinfo}
 
    @app.post("/v1/kapi/inpaint")
    async def processInpaint(data: apiInputPlusImage):
        buffer = BytesIO(base64.b64decode(data['initimage']['image']))
        initimg = Image.open(buffer)
        buffer = BytesIO(base64.b64decode(data['initimage']['mask']))
        initmask = Image.open(buffer)
        switch_mode = 1
        oimages, oinfo, ohtml = modules.img2img.img2img(data['prompt'],\
                                initimg,\
                                {'image':initimg, 'mask':initmask},\
                                data['steps'],\
                                smp_index,\
                                data['mask_blur'],\
                                data['inpainting_fill'],\
                                data['use_gfpgan'],\
                                False,\
                                switch_mode, \
                                data['batch_count'], \
                                1, \
                                data['cfg_scale'],\
                                data['denoising_strength'],\
                                data['seed'],\
                                data['height'],\
                                data['width'],\
                                data['resize_mode'],\
                                data['upscaler'],\
                                data['upscale_overlap'],\
                                data['inpaint_full_res'],\
                                data['inpainting_mask_invert'],\
                                0)

        b64images = []
        for img in oimages:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            b64images.append(img_str.decode())

        return {'images':b64images,'info':oinfo}

   # the fastapi server is already there, we want to block and wait on a ctrl+c or other error
    try:
        while True:
          time.sleep(0.1)
    except (KeyboardInterrupt, OSError):
        print("Keyboard interruption in main thread... closing server.")
        app.close()



if __name__ == "__main__":
    webui()
