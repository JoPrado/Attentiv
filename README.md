# Attentiv – AI-Powered Remote Attention Detection

**Attentiv** is an AI-driven focus monitoring tool that uses computer vision to assess attention levels in remote workers through facial cues. It provides real-time feedback on user focus to help teams optimize productivity in hybrid and remote environments.

---

## Overview

As remote work becomes a standard across industries, measuring productivity effectively has become increasingly difficult. Existing tools monitor device activity, but fail to assess actual engagement. Attentiv offers a more meaningful approach by analyzing webcam-based visual indicators—such as yawning and eye closure—to calculate a focus score in real time.

---

## Features

- Real-time facial attention analysis via webcam
- Focus scoring based on instances of distraction
- Non-invasive: no recording or image storage
- Lightweight deployment with frame-skipping strategy
- Output reports with focus and distraction metrics

---

## Technical Overview

- **Model**: CNN trained on a labeled dataset of facial states (`yawn`, `eyes closed`, etc.)
- **Deployment**: OpenCV + Caffe-based face detection
- **Optimization**: Frame-skipping to reduce CPU usage
- **Privacy**: No image storage; local inference only

---

## Installation

> Requires Python 3.8+ and a webcam

```bash
git clone https://github.com/your-org/attentiv.git
cd attentiv
pip install -r requirements.txt
```

---

##Training the Model

To train the CNN model on the dataset:

```bash
python train_model.py
```

> The trained model will be saved as `model.h5` in the `/models` directory.

---

##Running the Attention Detection

Once the model is trained (or if you're using a pre-trained one):

```bash
python deploy_model.py
```

> This will start the webcam and display live focus status. A summary report will be generated in `/outputs`.

---

##Project Structure

```bash
attentiv/
│
├── train_model.py         # Training logic
├── deploy_model.py        # Deployment logic (real-time webcam input)
├── models/
│   └── model.h5           # Trained model
├── outputs/
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## Roadmap

- Collect proprietary WFH dataset for improved accuracy
- Build GUI for real-time feedback and easier use
- Manager dashboard for aggregated focus insights
- Add additional indicators like posture and eye aspect ratio
- Implement encrypted communication and tamper detection

---

## License

This project is licensed under the MIT License.

---

## Contact

For inquiries, partnerships, or integration:
**Email:** vpereirafial.ieu2022@student.ie.edu & jtobar.ieu2022@student.ie.edu 
