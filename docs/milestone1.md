# CSC 425 AI Project — Milestone 1

## Project Title
**Bye-Bye Lanternfly: Using an AI Image Recognition Model to Identify and Count Lanternflies in Any Given Picture**

## Team
**Mackenzie Glaser** — Project lead; dataset preparation, preprocessing, model development, evaluation, counting, documentation, and presentation.

## Problem
Spotted lanternflies are an invasive species that can harm ecosystems and agriculture. The project proposes an image-recognition tool that helps users identify lanternflies and count how many appear in a photograph.

### Input → AI Task → Output

**Input:** User-provided image

**AI Task:** Object detection using YOLO

**Output:** Bounding boxes, confidence scores, and total number of detected lanternflies.

## Data
The initial goal is approximately 500 labeled images from open-source resources such as Hugging Face, iNaturalist, and Kaggle, subject to licensing and annotation availability.

Preprocessing may include resizing, augmentation, train/validation/test splitting, duplicate checking, and bounding-box annotation.

## Proposed AI
YOLOv8 will be used as the initial object detector. Training is planned in Google Colab using a free GPU. Counting is performed by counting the detected bounding boxes.

## Related Work
See the references in the project README. The project combines computer-vision literature with spotted-lanternfly biology/ecology literature.

## Timeline
- Now–Oct. 11: project overview and GitHub setup
- Oct. 11–25: data preparation and baseline
- Oct. 26–Nov. 15: AI approach and error analysis
- Nov. 16–29: system refinement and counting evaluation
- Nov. 30–Dec. 11: report, demo, and presentation

## Expected Deliverable
A lightweight trained `.pt` model, Python demo/inference program, GitHub repository, setup instructions, final report, and presentation.
