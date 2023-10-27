from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import uuid
from io import BytesIO
from datetime import datetime
import argparse


from threading import Lock

imageUpscaleLock = Lock()

app = Flask(__name__)


from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_img2img")

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to(torch_device="cuda", torch_dtype=torch.float16)
pipe.safety_checker=None

def upscale(input_images,prompt="high resolution photograph",strength=0.5,num_inference_steps=2):
    prompts=[prompt]*len(input_images)
    images = pipe(prompt=prompts, image=input_images, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images
    return images


def upscale_and_save(images, filename_prefix):
    #width, height = images[0].size
    to_uspcale=[]
    for image in images:
        # Upscale and save image, return new filename
        
        image = image.resize((512,512), Image.BICUBIC)
        to_uspcale.append(image)

    upscaled_images = upscale(to_uspcale,strength=args.strength,prompt=args.prompt,num_inference_steps=args.num_inference_steps)

    image_paths=[]

    for image in upscaled_images:
    
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        new_filename = f"{filename_prefix}_{current_time}_{uuid.uuid4().hex}.png"
        
        image_path = os.path.join("static/samples", new_filename)
        image.save(image_path)
        image_paths.append(image_path)
        
    return image_paths

@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files['file']
    img = Image.open(file.stream)

    #center crop image to make it square
    width, height = img.size
    if width > height:
        left = (width - height)/2
        right = (width + height)/2
        top = 0
        bottom = height
        img = img.crop((left, top, right, bottom))
    elif height > width:
        left = 0
        right = width
        top = (height - width)/2
        bottom = (height + width)/2
        img = img.crop((left, top, right, bottom))

    #resize image to 512x512
    img = img.resize((512, 512), Image.BICUBIC)

    filename_prefix = uuid.uuid4().hex
    img_path = os.path.join("static/samples", f"{filename_prefix}.png")
    img.save(img_path)
    return jsonify({"id": filename_prefix,"path": img_path})

@app.route("/upscale", methods=["POST"])
def upscale_image():
    data = request.json
    image_id = data["id"]
    image_path = data["path"]
    image = Image.open(image_path)

    #first convert to rgb
    image = image.convert('RGB')
    
    # Break the image into 4 pieces and upscale
    tiles = []
    for j in range(2):
        for i in range(2):
            left = i * (image.width // 2)
            upper = j * (image.height // 2)
            right = left + (image.width // 2)
            lower = upper + (image.height // 2)
            
            tile = image.crop((left, upper, right, lower))
            #new_tile_path = upscale_and_save(tile, image_id)
            #tiles.append(new_tile_path)
            tiles.append(tile)

    with imageUpscaleLock:
        tiles = upscale_and_save(tiles, image_id)
    
    return jsonify({"tiles": [os.path.join("./static/samples", os.path.basename(tile)) for tile in tiles]})

@app.route("/")
def index():
    return render_template("zoom.html")

if __name__ == "__main__":
    if not os.path.exists("static/samples"):
        os.makedirs("static/samples")
    
    
    #pass in strength as an argumnet
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--strength", type=float, default=0.5)
    
    
    #prompt = "high resolution photograph"
    argparser.add_argument("--prompt", type=str, default="high resolution photograph")

    #num_inference_steps = 2
    argparser.add_argument("--num_inference_steps", type=int, default=2)
    
    args = argparser.parse_args()
    
    
    app.run(debug=True, use_reloader=False)
