
for scene in 'Scene01' 'Scene02' 'Scene06' 'Scene18' 'Scene20'
do 
    # 生成 pose 得到 rmse 结果 - Generate pose and get rmse result
    python VO_Module/evaluation_scripts/test_vo.py \
    --datapath=datasets/Virtual_KITTI2/$scene \
    --disable_vis --segm_filter True

    # 逐帧生成 flow 和 depth - Generate flow and depth frame by frame
    python VO_Module/evaluation_scripts/test_vo2.py --scene $scene
done
