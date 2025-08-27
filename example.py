from scripts import image_quote_generator as img_q
# Use: Put your python scripts in the currently directory and execute -- with darts_utils.py

# run : python ./example.py
img_q.generate_quote_images(
    quotes=["Hello everybody!"],
    image_paths=["./utils/bg1.jpg", "./utils/bg2.jpg"],
    output_folder="output_images",
    font_path="./utils/Roboto-Black.ttf"
)