import cv2
import torch
from torchvision import transforms
from src.models.cnn import EmotionCNN
from src.models.vgg16_transfer import get_vgg16
from PIL import Image
import time

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_model(model_type='cnn', model_path='best_model.pth', device='cpu'):
    if model_type == 'cnn':
        model = EmotionCNN(num_classes=7)
    else:
        model = get_vgg16(num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_face(face_img):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    face_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return transform(face_img).unsqueeze(0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_type='cnn', device=device)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_tensor = preprocess_face(face_img).to(device)
            start = time.time()
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
            end = time.time()
            label = EMOTION_LABELS[pred.item()]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} ({(end-start)*1000:.1f}ms)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
