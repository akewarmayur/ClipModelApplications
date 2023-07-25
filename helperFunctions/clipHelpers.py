import clip
import torch
from PIL import Image
from itertools import islice


class ClipHelpers:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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