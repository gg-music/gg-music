ffmpeg -framerate 480 -pattern_type glob -i '/home/gtzan/jimmy/logs/cyclegan/model1/guitar_to_piano/*.png' -c:v libx264 -pix_fmt yuv420p /home/gtzan/jimmy/logs/cyclegan/guitar_to_piano.mp4
ffmpeg -framerate 480 -pattern_type glob -i '/home/gtzan/jimmy/logs/cyclegan/model1/piano_to_guitar/*.png' -c:v libx264 -pix_fmt yuv420p /home/gtzan/jimmy/logs/cyclegan/piano_to_guitar.mp4
