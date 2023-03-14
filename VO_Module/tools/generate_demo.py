import PIL.Image as Image
import os
import numpy as np
import cv2
import imageio

pred_dir = 'tools/edit-demo/results/Scene02/warp_img'
filelist = os.listdir(pred_dir)
filelist.sort()

# img_dir = "../datasets/cityscapes_vps/val/img_all"
# img_dir = "../datasets/viper/val/img"
# imagelist = os.listdir(img_dir)
# imagelist.sort()

out_folder = "tools/edit-demo/video"
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

cnt = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("3.avi", fourcc, 8, (1240,368))
for img_file in filelist:
    save_path = os.path.join(out_folder, img_file)

    img = cv2.imread(os.path.join(pred_dir, img_file))
    # background = cv2.imread(os.path.join(img_dir, img_file))

    # added_image = cv2.addWeighted(background,0.5,overlay,0.5, 0)

    
    # cv2.imwrite(save_path, added_image)
    # img = cv2.imread(save_path)
    video.write(img)

video.release()
