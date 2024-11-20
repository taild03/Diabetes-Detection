# Diabetes Risk Detection Application

A modern, user-friendly desktop application for assessing diabetes risk based on various health metrics. The application uses machine learning to provide risk assessments and personalized recommendations.

![image](https://github.com/user-attachments/assets/30f37160-218e-482e-9471-b568c9ef0fdc)


## Features

- 🎯 Real-time diabetes risk assessment
- 💉 Comprehensive health metrics analysis
- 🎨 Modern, intuitive user interface
- 📊 Machine learning-powered predictions
- 💡 Personalized health recommendations
- 📝 Detailed input field descriptions and guidelines

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - tkinter
  - pandas
  - numpy
  - scikit-learn
  - joblib

## Installation

1. Clone this repository or download the source code:
```bash
git clone https://github.com/taild03/diabetes-risk-detection.git
cd diabetes-risk-detection
```

2. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

3. Ensure you have the diabetes dataset (`diabetes.csv`) in the same directory as the application.

## Usage

1. Run the application:
```bash
python diabetes_detection_app.py
```

2. Enter the following health metrics in the provided fields:
   - Number of pregnancies
   - Glucose level (mg/dL)
   - Blood pressure (mm Hg)
   - Skin thickness (mm)
   - Insulin level (mu U/ml)
   - BMI (Body Mass Index)
   - Diabetes pedigree function
   - Age

3. Click the "Calculate Risk" button to get your risk assessment.

## Technical Details

### Machine Learning Model

- Uses a Decision Tree Classifier from scikit-learn
- Features are scaled using StandardScaler
- Model is trained on the first run and saved for future use
- Achieves reliable accuracy on the test set

### UI Components

- Built with tkinter and ttk for a modern look
- Custom styling with ModernStyle class
- Responsive design with proper spacing and layout
- Custom RoundedEntry widget for improved aesthetics

## File Structure

```
diabetes-risk-detection/
│
├── diabetes_detection_app.py    # Main application file
├── diabetes.csv                 # Dataset file
├── diabetes_model.joblib        # Saved model file (generated on first run)
├── diabetes_scaler.joblib       # Saved scaler file (generated on first run)
└── README.md                    # This file
```

## Logging

The application includes comprehensive logging functionality:
- Logs are generated with timestamps
- Model training and prediction events are logged
- Error handling with detailed logging

## Error Handling

- Input validation for all fields
- Graceful handling of missing or invalid data
- User-friendly error messages
- Proper exception handling for model training and predictions

## Health Guidelines

The application provides normal ranges for all health metrics:
- Glucose: 70-140 mg/dL
- Blood Pressure: 60-80 mm Hg
- Skin Thickness: 20-40 mm
- Insulin: 16-166 mu U/ml
- BMI: 18.5-24.9
- Diabetes Pedigree Function: 0.078-2.42

## Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider about your medical condition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: National Institute of Diabetes and Digestive and Kidney Diseases
- UI design inspired by modern material design principles
