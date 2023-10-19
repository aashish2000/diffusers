import os
import shutil
from PIL import Image

def resize_rename_images(source_path, resized_path, height, width):
    images_list = [x for x in os.listdir(source_path) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    resized_img= None
    if(not os.path.isdir(resized_path)):
        os.mkdir(resized_path)

    for img in images_list:
        load_img = Image.open(source_path + img).convert('RGB')

        if(load_img.size[1] >= load_img.size[0]):
            wpercent = (width/float(load_img.size[0]))
            hsize = int((float(load_img.size[1])*float(wpercent)))
            resized_img = load_img.resize((width,hsize), Image.LANCZOS)
            # resized_img = imutils.resize(load_img, width = width, inter = cv2.INTER_LANCZOS4)

        else:
            hpercent = (height/float(load_img.size[1]))
            wsize = int((float(load_img.size[0])*float(hpercent)))
            resized_img = load_img.resize((wsize, height), Image.LANCZOS)
            # resized_img = imutils.resize(load_img, height = height, inter = cv2.INTER_LANCZOS4)
        
        resized_img.save(resized_path + img.split(".")[0].split("_")[0] + ".jpg")

# file_arrays = {}
# for folders in os.listdir("./outputs/"):
#     file_arrays[folders] = sorted(os.listdir("./outputs/" + folders + "/test/"))[:500]

# os.mkdir("./outputs/metrics_test/")
# for key in file_arrays:
#     os.mkdir("./outputs/metrics_test/" + key)
#     for val in file_arrays[key]:
#         shutil.copyfile("./outputs/" + key + "/test/" + val, "./outputs/metrics_test/" + key + "/" + val)


# os.mkdir("./outputs/metrics_test/orig")
# for files in sorted(os.listdir("../../../../neurips/methods/diffusers/examples/text_to_image/outputs/lora_sd_v1_5/orig/"))[:500]:
#     shutil.copyfile("../../../../neurips/methods/diffusers/examples/text_to_image/outputs/lora_sd_v1_5/orig/" + files, "./outputs/metrics_test/orig/" + files)

# os.mkdir("./outputs/metrics_test/source")
# for files in sorted(os.listdir("../../../../neurips/datasets/non_entity_datasets/anna_ne_512/test/"))[:1000]:
#     shutil.copyfile("../../../../neurips/datasets/non_entity_datasets/anna_ne_512/test/" + files, "./outputs/metrics_test/source/" + files)

# os.mkdir("./outputs/metrics_test/sharpened")
# for files in sorted(os.listdir("../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_test/visualization/custom/"))[:500]:
#     shutil.copyfile("../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_test/visualization/custom/" + files, "./outputs/metrics_test/sharpened/" + files)

# resize_images("./outputs/metrics_test/sharpened/", "./outputs/metrics_test/sharpened/", 512, 512)

# for files in os.listdir("./outputs/metrics_test/sharpened/"):
#     file_name = files.split("_")[0]
#     os.rename("./outputs/metrics_test/sharpened/" + files, "./outputs/metrics_test/sharpened/" + file_name + ".jpg")

# for files in [x for x in os.listdir("./outputs/metrics_test/source/") if x.endswith(".txt")]:
#     shutil.copyfile("./outputs/metrics_test/source/" + files, "./outputs/metrics_test/sharpened/" + files)
    
# print(sorted(os.listdir("./outputs/metrics_test/sharpened/")) == sorted(os.listdir("./outputs/metrics_test/source/")))
# for files in sorted(os.listdir("./outputs/metrics_test/sharpened/")):
#     if(not os.path.isfile("./outputs/metrics_test/source/" + files)):
#         print(files)

# print(len(sorted(os.listdir("./outputs/metrics_test/sharpened/"))) == len(sorted(os.listdir("./outputs/metrics_test/source/"))))


# for files in sorted(os.listdir("../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_test/visualization/custom/"))[:500]:
#     shutil.copyfile("../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_test/visualization/custom/" + files, "./outputs/metrics_test/sharpened/" + files)

# resize_rename_images("../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_val/visualization/custom/", "../../../../neurips/datasets/non_entity_datasets/anna_ne_sharpened_512/val/", 512, 512)
# resize_rename_images("../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_train2/visualization/custom/", "../../../../neurips/methods/HAT/results/train2_resized/", 512, 512)

# for files in [x for x in os.listdir("../../../../neurips/datasets/non_entity_datasets/anna_ne_512/train/") if x.endswith(".jpg")]:
#     if(not os.path.isfile("../../../../neurips/datasets/non_entity_datasets/anna_ne_sharpened_512/train/" + files)):
#         shutil.copyfile("../../../../neurips/datasets/non_entity_datasets/anna_ne_512/train/" + files, "../../../../neurips/methods/HAT/datasets/anna_ne_512_remaining/" + files)

a = set([x for x in os.listdir("../../../../neurips/datasets/non_entity_datasets/anna_ne_caption_prefixes/objects_list/train/") if x.endswith(".jpg")])
b = set([x for x in os.listdir("../../../../neurips/datasets/non_entity_datasets/anna_ne_caption_prefixes/objects_list/huggingface/train/") if x.endswith(".jpg")])

print(len(a.difference(b)), len(a), len(b))
print(b.difference(a))

# for files in [x for x in os.listdir("../../../../neurips/datasets/non_entity_datasets/anna_ne_sharpened_512/train/") if x.endswith(".jpg")]:
#     if(files.endswith("usa.jpg")):
#         os.rename("../../../../neurips/datasets/non_entity_datasets/anna_ne_sharpened_512/train/" + files, "../../../../neurips/datasets/non_entity_datasets/anna_ne_sharpened_512/train/" + files.split(".")[0] + "_today.jpg")
    