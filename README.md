# machine-vision
Our goal is to be able to read text from a phone screen.

run live camera pipeline with:
python live_feed_screen_reader.py

camera_move runs glare_detection + screen_detection
glare_detection runs move_instructions

camera_move outputs saved image to saved_images/detected_screen
