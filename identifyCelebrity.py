import clip
import torch
from PIL import Image, ImageDraw, ImageFont
from itertools import islice
import glob
from prompts import celebrityList
from helperFunctions.detectFaces import FaceDetection
import re
import pandas as pd


class Celebrity:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def add_text_to_image(self, image, text, position=(20, 20), font_size=15, font_color=(255, 255, 255)):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text(position, text, font=font, fill=font_color)

    # Function to resize images to a specified size
    def resize_image(self, image_path, target_size):
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        return image

    def create_collage(self, images, texts, canvas_size=(800, 600), image_size=(100, 100)):
        canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))

        x_offset = 0
        for i, (image_path, text) in enumerate(zip(images, texts)):
            image = self.resize_image(image_path, image_size)
            canvas.paste(image, (x_offset, 0))
            self.add_text_to_image(canvas, text, position=(x_offset + 5, 5))
            x_offset += image_size[0] + 5

        return canvas

    def get_clip_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def get_prediction(self, frame_path, list_of_labels, how_many_predictions, model, preprocess) -> list:
        Highest3Predictions = []
        try:
            text = clip.tokenize(list_of_labels).to(self.device)
            image = preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs = probs.tolist()[0]
            vv = {}
            for i, j in enumerate(probs):
                vv[list_of_labels[i]] = j
            maxx = {k: v for k, v in sorted(vv.items(), key=lambda item: item[1], reverse=True)}
            Highest3Predictions = list(islice(maxx.items(), how_many_predictions))
            print(f"{frame_path} : {Highest3Predictions}")
        except Exception as e:
            print("Exception in CLIP predictions:", e)

        return Highest3Predictions

    def startProcess(self, images_path):
        objFD = FaceDetection()
        model, preprocess = self.get_clip_model()
        list_of_prompts = []
        for celebrities in celebrityList.celebrity_list:
            list_of_prompts.append("a photo of " + str(celebrities))
        list_of_prompts.append("a photo of other person")
        list_of_prompts.append("a photo of ")
        faces_path = []
        detected_celebrities = []
        results = pd.DataFrame(columns=["FrameFileName", "FacePath", "CelebrityName"])
        try:
            list_of_images = []
            isFolder = True
            tm = images_path.split(".")
            if len(tm) > 1:
                isFolder = False
            if isFolder:
                for fi in glob.glob(images_path + "/*"):
                    list_of_images.append(fi)
            else:
                list_of_images.append(images_path)
            list_of_images.sort(key=self.natural_keys)
            extract_faces_df = objFD.extractFaces(list_of_images)
            extract_faces_df.to_csv("facesInfo.csv")

            for ind, row in extract_faces_df.iterrows():
                face_path = row['PaddedFacesPath']
                Highest3Predictions = self.get_prediction(face_path, list_of_prompts,
                                                          3, model, preprocess)
                c1 = Highest3Predictions[0][0]
                s1 = round(100 * Highest3Predictions[0][1], 2)
                if s1 > 70:
                    detected_celebrity = c1.split("a photo of ")[1]
                    df_length1 = len(results)
                    results.loc[df_length1] = [row['FrameFileName'], face_path, detected_celebrity]
                    if len(face_path) < 100:
                        faces_path.append(face_path)
                        detected_celebrities.append(detected_celebrity)
            collage = self.create_collage(faces_path, detected_celebrities)

            # Save the collage
            collage_path = "Results/collage.jpg"
            collage.save(collage_path)
            collage.show()
            results.to_csv("Results/CelebrityResult.csv")
        except Exception as e:
            print(e)
            raise


obj = Celebrity()
obj.startProcess("images")
