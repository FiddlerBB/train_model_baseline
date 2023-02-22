import os 
from glob import glob
import cv2 
import asyncio

IMAGES_PATH = r'data/data_face/*'
OUTPUT_PATH = r'data/data_face_224/'

async def resize_224(img_path):
    image = cv2.imread(img_path)
    resized = cv2.resize(image, (224,224))
    return resized

async def process_image(input, output):
    loop = asyncio.get_running_loop()
    image_base_name = os.path.basename(input)
    resized = await resize_224(input)
    output_path = os.path.join(output, image_base_name)
    await loop.run_in_executor(None, cv2.imwrite, output_path, resized)

async def main():
    tasks = []
    for image in glob(IMAGES_PATH):
        task = asyncio.create_task(process_image(image, OUTPUT_PATH))
        tasks.append(task)

    await asyncio.gather(*tasks)

asyncio.run(main())