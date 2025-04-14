# Happy vs. Sad Emotion Classifier 😄😢

This is my first AI project, built entirely from scratch using PyTorch.  
The model classifies 48×48 grayscale face images as either **Happy** or **Sad**, based on the FER-2013 dataset.

I trained and deployed this model myself as part of a self-directed AI engineering roadmap — no pre-trained models, no convolutional layers, just fundamentals, tuning, and iteration.

## Model Details

- Accuracy: ~75%
- 4-layer fully connected network (2304 → 1152 → 576 → 192 → 2)
- Custom dataset split from FER-2013 (happy/sad only)
- ⚙Normalization and manual balancing (no augmentation)
- Trained locally on GTX 2060
- Deployed using [Gradio](https://gradio.app) + [Hugging Face Spaces](https://huggingface.co/spaces/AljazR/happy-sad-emotion-classifier)

## Try It Live

👉 [Click to open the app](https://aljazr-happy-sad-emotion-classifier.hf.space)

## Files

- `app.py`: Gradio interface
- `emotion_model.pth`: Trained PyTorch model
- `requirements.txt`: Dependencies

## About Me

I'm currently working through a custom AI Engineering roadmap.  
This was my first showcase project — many more coming soon!
