ffmpeg -framerate 24 -pattern_type glob -i '/home/gtzan/jimmy/logs/cyclegan/*.png' -c:v libx264 -pix_fmt yuv420p /home/gtzan/jimmy/logs/cyclegan/output.mp4
