import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModernStyle:
    # Enhanced color scheme
    PRIMARY_COLOR = "#1E88E5"  # Medium blue
    SECONDARY_COLOR = "#43A047"  # Fresh green
    BACKGROUND_COLOR = "#E3F2FD"  # Light blue background
    CARD_BG = "#F8F9FA"  # Soft gray for cards
    TEXT_COLOR = "#212121"  # Dark gray
    
    # Fonts
    TITLE_FONT = ('Helvetica', 24, 'bold')

    @classmethod
    def apply_style(cls):
        style = ttk.Style()
        
        # Get alt theme button settings
        alt_style = ttk.Style()
        alt_style.theme_use('alt')
        btn_settings = alt_style.lookup('TButton', 'relief')
        btn_map = alt_style.map('TButton')

        # Reset to default theme
        style.theme_use('default')

        # Configure frame styles
        style.configure('Modern.TFrame', background=cls.BACKGROUND_COLOR)
        style.configure('Card.TFrame', 
                       background=cls.CARD_BG,
                       relief='solid',
                       borderwidth=1)
        
        
        # Configure label styles
        style.configure('Modern.TLabel', 
                       background=cls.BACKGROUND_COLOR, 
                       foreground=cls.TEXT_COLOR)
        style.configure('Card.TLabel',
                       background=cls.CARD_BG,
                       foreground=cls.TEXT_COLOR)
        style.configure('Title.TLabel',
                       background=cls.BACKGROUND_COLOR,
                       foreground=cls.PRIMARY_COLOR,
                       font=cls.TITLE_FONT)
        
        # Configure custom button style
        style.configure('Alt.TButton', relief=btn_settings,
                       background=cls.PRIMARY_COLOR,
                       foreground='white',
                       padding=(40, 15),
                       font=('Helvetica', 13, 'bold'))
        style.map('Alt.TButton', **dict(btn_map),
                 background=[('active', cls.SECONDARY_COLOR), ('!disabled', cls.PRIMARY_COLOR)])

class RoundedEntry(tk.Canvas):
    def __init__(self, parent, width=300, height=35, *args, **kwargs):
        tk.Canvas.__init__(self, parent, width=width, height=height, 
                          bg=ModernStyle.BACKGROUND_COLOR, highlightthickness=0)
        
        # Create rounded rectangle
        self.rect = self.round_rectangle(2, 2, width-2, height-2, 
                                       radius=8, fill='white')
        
        # Create entry widget
        self.entry = tk.Entry(self, border=0, justify='left',
                            bg='white', font=('Helvetica', 12))
        self.entry.place(x=10, y=height//2, anchor='w')
        self.entry.configure(width=width//9)  # Adjust entry width

    def round_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                 x1+radius, y1,
                 x2-radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1+radius,
                 x1, y1]

        return self.create_polygon(points, smooth=True, **kwargs)

    def get(self):
        return self.entry.get()

class DiabetesDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Risk Detection")
        self.root.geometry("1200x825")
        self.root.configure(bg=ModernStyle.BACKGROUND_COLOR)
        
        # Center the window
        self.center_window()
        
        # Apply modern styling
        ModernStyle.apply_style()
        
        # Input field descriptions
        self.field_descriptions = {
            "Pregnancies": "Number of times pregnant. \nEnter 0 if male or never pregnant.",
            "Glucose": "Plasma glucose concentration (mg/dL). \nNormal: 70-140 mg/dL.",
            "BloodPressure": "Diastolic blood pressure (mm Hg). \nNormal: 60-80 mm Hg.",
            "SkinThickness": "Triceps skin fold thickness (mm). \nNormal: 20-40 mm.",
            "Insulin": "2-Hour serum insulin (mu U/ml). \nNormal: 16-166 mu U/ml.",
            "BMI": "Body Mass Index = weight(kg)/(height(m))². \nNormal: 18.5-24.9.",
            "DiabetesPedigreeFunction": "Diabetes family history function. \nRange: 0.078-2.42.",
            "Age": "Age in years. \nMust be greater than 0."
        }
        
        # Create main container
        self.create_main_container()
        
        # Load or train model
        self.load_or_train_model()

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1200
        window_height = 825
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_main_container(self):
        # Main frame
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(expand=True, fill='both', padx=40, pady=20)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Diabetes Risk Assessment",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 10))

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Please fill all the input fields",
            style='Modern.TLabel'
        )
        subtitle_label.pack(pady=(3, 3))

        # Create frame for input fields
        fields_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        fields_frame.pack(fill='both', expand=True)

        # Create left and right columns
        left_frame = ttk.Frame(fields_frame, style='Modern.TFrame')
        right_frame = ttk.Frame(fields_frame, style='Modern.TFrame')
        left_frame.pack(side='left', fill='both', expand=True, padx=20)
        right_frame.pack(side='right', fill='both', expand=True, padx=20)

        # Create input fields in two columns
        self.inputs = {}
        field_names = list(self.field_descriptions.keys())
        mid_point = len(field_names) // 2

        for i, field in enumerate(field_names):
            parent_frame = left_frame if i < mid_point else right_frame
            self.create_input_field(parent_frame, field)

        # Calculate button with modern style
        button_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        button_frame.pack(fill='x', pady=30)
        
        calculate_btn = ttk.Button(
            button_frame,
            text="Calculate Risk",
            command=self.calculate_risk,
            style='Alt.TButton'
        )
        calculate_btn.pack()

        # Result label
        self.result_var = tk.StringVar()
        result_label = ttk.Label(
            main_frame,
            textvariable=self.result_var,
            wraplength=800,
            style='Modern.TLabel',
            font=('Helvetica', 12)
        )
        result_label.pack(pady=20)

    def create_input_field(self, parent, field):
        # Card frame for each input group
        card_frame = ttk.Frame(parent, style='Card.TFrame')
        card_frame.pack(fill='x', pady=10, padx=5, ipady=10)

        # Inner padding frame
        padding_frame = ttk.Frame(card_frame, style='Card.TFrame')
        padding_frame.pack(fill='x', padx=15, pady=5)

        # Label
        label = ttk.Label(
            padding_frame,
            text=f"{field}:",
            style='Card.TLabel',
            font=('Helvetica', 12, 'bold')
        )
        label.pack(anchor='w')

        # Description
        desc_label = ttk.Label(
            padding_frame,
            text=self.field_descriptions[field],
            wraplength=400,
            style='Card.TLabel',
            font=('Helvetica', 10, 'italic')
        )
        desc_label.pack(anchor='w', pady=(0, 5))

        # Custom rounded entry
        entry = RoundedEntry(padding_frame)
        entry.pack(anchor='w', pady=(5, 0))
        self.inputs[field] = entry

    def load_or_train_model(self):
        model_path = 'diabetes_model.joblib'
        scaler_path = 'diabetes_scaler.joblib'

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logging.info("Loading existing model and scaler...")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = list(self.field_descriptions.keys())
            logging.info("Model and scaler loaded successfully")
        else:
            logging.info("Training new model...")
            self.train_model()
            # Save the model and scaler
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logging.info("Model and scaler saved for future use")

    def train_model(self):
        try:
            logging.info("Loading dataset...")
            data = pd.read_csv('diabetes.csv')
            
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            
            logging.info("Splitting dataset...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            logging.info("Scaling features...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            logging.info("Training model...")
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
            self.model.fit(X_train_scaled, y_train)
            
            self.feature_names = X.columns.tolist()
            
            # Test accuracy
            X_test_scaled = self.scaler.transform(X_test)
            accuracy = self.model.score(X_test_scaled, y_test)
            logging.info(f"Model training complete. Test accuracy: {accuracy:.2%}")

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            messagebox.showerror("Error", "Failed to train model. Please check the logs.")

    def calculate_risk(self):
        try:
            input_values = []
            for feature in self.feature_names:
                value = float(self.inputs[feature].get())
                input_values.append(value)
            
            input_scaled = self.scaler.transform([input_values])
            prediction = self.model.predict(input_scaled)
            probability = self.model.predict_proba(input_scaled)[0]
            
            if prediction[0] == 1:
                risk_level = f"High Risk ({probability[1]:.1%})"
                message = (
                    "❌Warning!❌ Based on the provided information, you may have "
                    "an elevated risk of diabetes. Please consult a healthcare "
                    "professional for proper medical evaluation.\n\n"
                    "Recommended actions:\n"
                    "1. Schedule an appointment with your doctor\n"
                    "2. Monitor your blood sugar levels\n"
                    "3. Maintain a healthy diet\n"
                    "4. Exercise regularly"
                )
                self.result_var.set(f"Result: {risk_level}\n\n{message}")
                logging.info("High risk prediction made")
                messagebox.showinfo("Diabetes Risk Result", f"Result: {risk_level}\n\n{message}")

            else:
                risk_level = f"Low Risk ({probability[0]:.1%})"
                message = (
                    "✔Good news!✔ Based on the provided information, you appear "
                    "to have a lower risk of diabetes. However, maintaining a "
                    "healthy lifestyle is always important.\n\n"
                    "Recommendations:\n"
                    "1. Continue regular health check-ups\n"
                    "2. Maintain a balanced diet\n"
                    "3. Stay physically active\n"
                    "4. Monitor your health regularly"
                )
                self.result_var.set(f"Result: {risk_level}\n\n{message}")
                logging.info("Low risk prediction made")
                messagebox.showinfo("Diabetes Risk Result", f"Result: {risk_level}\n\n{message}")

        except ValueError:
            messagebox.showerror(
                "Error", 
                "Please ensure all fields contain valid numerical values."
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesDetectionApp(root)
    root.mainloop()