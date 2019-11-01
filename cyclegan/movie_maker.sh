model=$1
MODEL_ROOT=/home/gtzan/ssd/models/$model
python -m cyclegan.plot_history -m $model
mkdir $MODEL_ROOT/mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/fake_x/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/fake_x.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/fake_y/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/fake_y.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/disc_fake_x/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/disc_fake_x.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/disc_fake_y/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/disc_fake_y.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/disc_real_x/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/disc_real_x.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/disc_real_y/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/disc_real_y.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/Generator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/Generator_loss.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i $MODEL_ROOT'/images/Discriminator_loss/*.png' -c:v libx264 -pix_fmt yuv420p $MODEL_ROOT/mp4/Discriminator_loss.mp4
ffmpeg -y -i $MODEL_ROOT/mp4/fake_x.mp4 -i $MODEL_ROOT/mp4/fake_y.mp4 \
-i $MODEL_ROOT/mp4/disc_fake_x.mp4 -i $MODEL_ROOT/mp4/disc_fake_y.mp4 \
-i $MODEL_ROOT/mp4/disc_real_x.mp4 -i $MODEL_ROOT/mp4/disc_real_y.mp4 \
-i $MODEL_ROOT/mp4/Generator_loss.mp4 -i $MODEL_ROOT/mp4/Discriminator_loss.mp4 \
-filter_complex "[2:v][4:v]vstack=inputs=2[dx];[3:v][5:v]vstack=inputs=2[dy];[0:v][1:v][6:v]hstack=inputs=3[top];[dx][dy][7:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[v]" \
-map "[v]" $MODEL_ROOT/mp4/${model}_history.mp4
