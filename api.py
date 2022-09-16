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
import modules.gfpgan_model as gfpgan
import modules.realesrgan_model as realesrgan
#import modules.esrgan_model as esrgan
import modules.images as images
import modules.lowvram
from modules.txt2img import *
from modules.img2img import *
from urllib import request
from flask import Flask, Response, request, send_file, jsonify
from flask_ngrok import run_with_ngrok
import json
from io import BytesIO
import base64
from modules.sd_samplers import samplers, samplers_for_img2img,samplers_k_diffusion
import gdiffusion
import PIL
from modules.realesrgan_model import RealesrganModelInfo

def api(): 
    shared.sd_upscalers = [
        RealesrganModelInfo(
            name="RealESRGAN",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            netscale=4, model=lambda img: realesrgan.upscale_with_realesrgan(img, 2, 0)
        ),
        RealesrganModelInfo(
            name="Lanczos",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            netscale=4, model=lambda img: img.resize((img.width*2, img.height*2), resample=images.LANCZOS)
        ),
        RealesrganModelInfo(
            name="None",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            netscale=2, model=lambda img: img
        ),
    ]
    #esrgan.load_models(cmd_opts.esrgan_models_path)
    realesrgan.setup_realesrgan()
    gfpgan.setup_gfpgan()


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

    def run_extras(image, gfpgan_strength, upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
        processing.torch_gc()

        image = image.convert("RGB")

        outpath = opts.outdir_samples or opts.outdir_extras_samples

        if gfpgan.have_gfpgan is not None and gfpgan_strength > 0:
            restored_img = gfpgan.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
            res = Image.fromarray(restored_img)

            if gfpgan_strength < 1.0:
                res = Image.blend(image, res, gfpgan_strength)

            image = res

        if upscaling_resize != 1.0:
            def upscale(image, scaler_index, resize):
                small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
                pixels = tuple(np.array(small).flatten().tolist())
                key = (resize, scaler_index, image.width, image.height) + pixels

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

    modules.scripts.load_scripts(os.path.join(script_path, "scripts"))


    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with singal {sig} in {frame}')
        os._exit(0)


    signal.signal(signal.SIGINT, sigint_handler)

    app = Flask(__name__)

    @app.route("/api/version")
    def processVersion():
        data={'version':2.0} 
        return jsonify(data)

    @app.route("/api/test")
    def processTest():
    #    oimages, oinfo, ohtml = txt2img(prompt= data['prompt'],steps= data['steps'],2,data['use_gfpgan'],data['batch_count'],1,data['cfg_scale'],data['seed'],data['height'],data['width'],0)
        b64images = []
        for img in oimages:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            b64images.append(img_str.decode())
        return jsonify({'images':b64images,'info':oinfo})



    @app.route("/api/", methods=["POST"])
    def processAPI():
        r = request
        data = json.loads(r.data)
        print(data['mode'])

        oimages = []
        oinfo = []
        smp_index=0
        for i in range(0,len(samplers_k_diffusion)):
            if samplers_k_diffusion[i][0]==data['sampler']: smp_index=i    

        if data['mode'] == 'txt2img':
            # only positional arguments allowed because of *args 
            oimages, oinfo, ohtml = txt2img(
                data['prompt'],"", "","",data['steps'], smp_index,
                data.get('restore_faces',False), data.get('tiling',False),
                data['batch_count'],1,data['cfg_scale'],data['seed'],
                0,0.0, 0, 0,
                data.get('height',512), data.get('width',512),
                0)
        if data['mode'] == 'img2img':
            switch_mode = 0
            buffer = BytesIO(base64.b64decode(data['initimage']['image']))
            initimg = Image.open(buffer)
            # only positional arguments allowed because of *args  (CSV)

    #def img2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, init_img, init_img_with_mask, init_mask, mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: #bool, mode: int, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, height: int, width: int, resize_mode: int, #upscaler_index: str, upscale_overlap: int, inpaint_full_res: bool, inpainting_mask_invert: int, *args):
        #  inpainting_fill:  fill', 'original', 'latent noise', 'latent nothing'
            oimages, oinfo, ohtml = img2img(data['prompt'],"","","",initimg, "","", 0, data.get("steps",15), 
                        smp_index, data.get("mask_blur",4),              # mode and blur
                        data.get('inpainting_fill',3), 
                        data.get('restore_faces',False), data.get('tiling',False),
                        switch_mode,        # 0=img2img
                        data.get('batch_count',1),1,
                        data.get('cfg_scale',7.5), data.get('denoising_strength',0.95),
                        data.get('seed',-1),
                        0,0.0, 0, 0,
                        data.get('height',512), data.get('width',512),
                        0, data.get("upscaler","RealESRGAN"), data.get("upscale_overlap",64),data.get("inpaint_full_res",True),data.get("inpainting_mask_invert",False),
                        0)

        if data['mode'] == 'inpainting':

            buffer = BytesIO(base64.b64decode(data['initimage']['image']))
            initimg = Image.open(buffer)
        # buffer = BytesIO(base64.b64decode(data['initimage']['mask']))
        # initmask = Image.open(buffer)
        
            fill_mode=int(data['inpainting_fill'])

            # experimental: if mode is g-diffusion switch to orginal and use g-diffusion image as init image
            # image as init_image
            if (fill_mode==4):
                if (512, 512) != initimg.size and fill_mode==4: # default size is native img size
                    print("Inpainting: Resizing input img to 512x512 ")    
                    initimg = initimg.resize((512, 512), resample=PIL.Image.LANCZOS)
                init_image, initmask=gdiffusion.get_init_image(initimg,0.99,1, 0)
                fill_mode=1 # original
                initimg=init_image            
            else:
                initmask=gdiffusion.getAlphaAsImage(initimg)    # mask is now generated on server
            if (not initmask):
                if (gdiffusion.maskError==1):
                    print("inpainting: No transparent pixels found - throwing error")
                    return jsonify({'error':1,'text':"No transparent pixels found in image"})
                else:
                    print("inpainting: No  pixels found - throwing error")
                    return jsonify({'error':2,'text':"No pixels found in image"})                

            switch_mode = 1
        
            oimages, oinfo, ohtml = img2img(data['prompt'],"","","",initimg, {'image':initimg, 'mask':initmask},
                        "", 0, data.get("steps",15), 
                        smp_index, data.get("mask_blur",4),              # mode and blur
                        data.get('inpainting_fill',3), 
                        data.get('restore_faces',False), data.get('tiling',False),
                        switch_mode,        # 0=img2img
                        data.get('batch_count',1),1,
                        data.get('cfg_scale',7.5), data.get('denoising_strength',0.95),
                        data.get('seed',-1),
                        0,0.0, 0, 0,
                        data.get('height',512), data.get('width',512),
                        0, data.get("upscaler","RealESRGAN"), data.get("upscale_overlap",64),data.get("inpaint_full_res",True),data.get("inpainting_mask_invert",False),
                        0)


        b64images = []
        for img in oimages:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            b64images.append(img_str.decode())

        return jsonify({'images':b64images,'info':oinfo})

    if cmd_opts.share: run_with_ngrok(app)
    app.run()
