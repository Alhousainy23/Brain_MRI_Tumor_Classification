# ============================================================
# Brain MRI Tumor Classification
# CustomTkinter GUI
#
# Models:
# 1. Baseline CNN
# 2. ResNet18
# 3. EfficientNet-B0
#
# Mode:
# Light Mode
# ============================================================

import sys
from pathlib import Path

# ============================================================
# 1. Add src Directory To Python Path
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 2. Imports
# ============================================================

import customtkinter as ctk

import torch
import pandas as pd

from PIL import Image

from tkinter import filedialog, messagebox

from torchvision import transforms


# ============================================================
# 3. Project Imports
# ============================================================

from models import BaselineCNN

from resnet_model import ResNet18Model

from efficientnet_model import EfficientNetB0Model

from config import (
    DEVICE,
    NUM_CLASSES,
    CLASS_NAMES,
    BASELINE_MODEL_PATH,
    RESNET18_MODEL_PATH,
    EFFICIENTNET_B0_MODEL_PATH,
    EVALUATION_DIR
)


# ============================================================
# 4. CustomTkinter Appearance
# ============================================================

ctk.set_appearance_mode("light")

ctk.set_default_color_theme("blue")


# ============================================================
# 5. Main Application
# ============================================================

class BrainTumorGUI(ctk.CTk):

    def __init__(self):

        super().__init__()

        # ====================================================
        # Window Configuration
        # ====================================================

        self.title(
            "Brain MRI Tumor Classification"
        )

        self.geometry(
            "1200x750"
        )

        self.minsize(
            1000,
            650
        )

        # ====================================================
        # Application State
        # ====================================================

        self.selected_image_path = None

        self.current_image = None

        self.current_photo = None

        self.current_model = None

        self.stop_requested = False

        self.loaded_models = {}

        # ====================================================
        # Image Transform
        # ====================================================

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )

        ])

        # ====================================================
        # Create GUI
        # ====================================================

        self.create_gui()

        # ====================================================
        # Load Best Model Information
        # ====================================================

        self.load_best_model_info()


    # ========================================================
    # Create GUI
    # ========================================================

    def create_gui(self):

        # ====================================================
        # Main Grid Configuration
        # ====================================================

        self.grid_columnconfigure(
            0,
            weight=1
        )

        self.grid_columnconfigure(
            1,
            weight=1
        )

        self.grid_rowconfigure(
            1,
            weight=1
        )

        # ====================================================
        # Header
        # ====================================================

        self.header_frame = ctk.CTkFrame(
            self,
            corner_radius=10
        )

        self.header_frame.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=20,
            pady=20,
            sticky="ew"
        )

        self.title_label = ctk.CTkLabel(

            self.header_frame,

            text=(
                "Brain MRI Tumor Classification"
            ),

            font=ctk.CTkFont(
                size=28,
                weight="bold"
            )
        )

        self.title_label.pack(
            pady=(15, 5)
        )

        self.subtitle_label = ctk.CTkLabel(

            self.header_frame,

            text=(
                "Baseline CNN  |  ResNet18  |  EfficientNet-B0"
            ),

            font=ctk.CTkFont(
                size=15
            )
        )

        self.subtitle_label.pack(
            pady=(0, 15)
        )


        # ====================================================
        # Left Frame
        # ====================================================

        self.left_frame = ctk.CTkFrame(
            self,
            corner_radius=10
        )

        self.left_frame.grid(
            row=1,
            column=0,
            padx=(20, 10),
            pady=(0, 20),
            sticky="nsew"
        )


        # ====================================================
        # Image Section
        # ====================================================

        self.image_title = ctk.CTkLabel(

            self.left_frame,

            text="MRI Image",

            font=ctk.CTkFont(
                size=20,
                weight="bold"
            )
        )

        self.image_title.pack(
            pady=(20, 10)
        )


        self.image_display = ctk.CTkLabel(

            self.left_frame,

            text=(
                "No Image Selected\n\n"
                "Click 'Load Image' to select MRI image"
            ),

            width=450,

            height=400,

            corner_radius=10,

            fg_color=(
                "#E5E7EB"
            ),

            text_color=(
                "#374151"
            )
        )

        self.image_display.pack(
            padx=20,
            pady=10,
            fill="both",
            expand=True
        )


        # ====================================================
        # Load Image Button
        # ====================================================

        self.load_button = ctk.CTkButton(

            self.left_frame,

            text="Load Image",

            height=45,

            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),

            command=self.load_image
        )

        self.load_button.pack(
            padx=20,
            pady=20,
            fill="x"
        )


        # ====================================================
        # Right Frame
        # ====================================================

        self.right_frame = ctk.CTkFrame(
            self,
            corner_radius=10
        )

        self.right_frame.grid(
            row=1,
            column=1,
            padx=(10, 20),
            pady=(0, 20),
            sticky="nsew"
        )


        # ====================================================
        # Model Selection
        # ====================================================

        self.model_title = ctk.CTkLabel(

            self.right_frame,

            text="Select Model",

            font=ctk.CTkFont(
                size=20,
                weight="bold"
            )
        )

        self.model_title.pack(
            pady=(25, 10)
        )


        self.model_combobox = ctk.CTkComboBox(

            self.right_frame,

            values=[
                "Baseline CNN",
                "ResNet18",
                "EfficientNet-B0"
            ],

            width=350,

            height=45,

            font=ctk.CTkFont(
                size=15
            ),

            command=self.on_model_selected
        )

        self.model_combobox.pack(
            padx=30,
            pady=10
        )

        self.model_combobox.set(
            "EfficientNet-B0"
        )


        # ====================================================
        # Predict Button
        # ====================================================

        self.predict_button = ctk.CTkButton(

            self.right_frame,

            text="Predict",

            width=350,

            height=50,

            font=ctk.CTkFont(
                size=17,
                weight="bold"
            ),

            command=self.predict_image
        )

        self.predict_button.pack(
            padx=30,
            pady=20
        )


        # ====================================================
        # Results Title
        # ====================================================

        self.results_title = ctk.CTkLabel(

            self.right_frame,

            text="Prediction Results",

            font=ctk.CTkFont(
                size=20,
                weight="bold"
            )
        )

        self.results_title.pack(
            pady=(10, 10)
        )


        # ====================================================
        # Result Frame
        # ====================================================

        self.result_frame = ctk.CTkFrame(

            self.right_frame,

            fg_color=(
                "#F3F4F6"
            ),

            corner_radius=10
        )

        self.result_frame.pack(
            padx=30,
            pady=10,
            fill="x"
        )


        # ====================================================
        # Selected Model Label
        # ====================================================

        self.selected_model_label = ctk.CTkLabel(

            self.result_frame,

            text="Selected Model: ---",

            font=ctk.CTkFont(
                size=15,
                weight="bold"
            )
        )

        self.selected_model_label.pack(
            pady=(15, 5)
        )


        # ====================================================
        # Prediction Label
        # ====================================================

        self.prediction_label = ctk.CTkLabel(

            self.result_frame,

            text="Prediction: ---",

            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        )

        self.prediction_label.pack(
            pady=5
        )


        # ====================================================
        # Confidence Label
        # ====================================================

        self.confidence_label = ctk.CTkLabel(

            self.result_frame,

            text="Confidence: ---",

            font=ctk.CTkFont(
                size=16
            )
        )

        self.confidence_label.pack(
            pady=(5, 15)
        )


        # ====================================================
        # Best Model Section
        # ====================================================

        self.best_model_title = ctk.CTkLabel(

            self.right_frame,

            text="Best Model",

            font=ctk.CTkFont(
                size=20,
                weight="bold"
            )
        )

        self.best_model_title.pack(
            pady=(20, 10)
        )


        self.best_model_label = ctk.CTkLabel(

            self.right_frame,

            text="Best Model: Loading...",

            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),

            fg_color=(
                "#DCFCE7"
            ),

            text_color=(
                "#166534"
            ),

            corner_radius=10,

            width=350,

            height=50
        )

        self.best_model_label.pack(
            padx=30,
            pady=10
        )


        # ====================================================
        # Bottom Buttons Frame
        # ====================================================

        self.buttons_frame = ctk.CTkFrame(

            self.right_frame,

            fg_color="transparent"
        )

        self.buttons_frame.pack(
            padx=30,
            pady=20,
            fill="x"
        )


        self.buttons_frame.grid_columnconfigure(
            0,
            weight=1
        )

        self.buttons_frame.grid_columnconfigure(
            1,
            weight=1
        )


        # ====================================================
        # Stop Button
        # ====================================================

        self.stop_button = ctk.CTkButton(

            self.buttons_frame,

            text="Stop",

            height=45,

            fg_color=(
                "#F59E0B"
            ),

            hover_color=(
                "#D97706"
            ),

            command=self.stop_prediction
        )

        self.stop_button.grid(

            row=0,

            column=0,

            padx=(0, 5),

            sticky="ew"
        )


        # ====================================================
        # Exit Button
        # ====================================================

        self.exit_button = ctk.CTkButton(

            self.buttons_frame,

            text="Exit",

            height=45,

            fg_color=(
                "#DC2626"
            ),

            hover_color=(
                "#B91C1C"
            ),

            command=self.exit_application
        )

        self.exit_button.grid(

            row=0,

            column=1,

            padx=(5, 0),

            sticky="ew"
        )


        # ====================================================
        # Status Bar
        # ====================================================

        self.status_label = ctk.CTkLabel(

            self,

            text="Status: Ready",

            anchor="w",

            font=ctk.CTkFont(
                size=13
            )
        )

        self.status_label.grid(

            row=2,

            column=0,

            columnspan=2,

            padx=25,

            pady=(0, 10),

            sticky="ew"
        )


    # ========================================================
    # Load Image
    # ========================================================

    def load_image(self):

        file_path = filedialog.askopenfilename(

            title="Select Brain MRI Image",

            filetypes=[

                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                ),

                (
                    "All Files",
                    "*.*"
                )

            ]
        )

        if not file_path:

            return


        try:

            self.selected_image_path = file_path

            self.current_image = Image.open(
                file_path
            ).convert(
                "RGB"
            )


            # ================================================
            # Display Image
            # ================================================

            display_image = self.current_image.copy()

            display_image.thumbnail(
                (450, 400)
            )


            self.current_photo = ctk.CTkImage(

                light_image=display_image,

                dark_image=display_image,

                size=display_image.size
            )


            self.image_display.configure(

                image=self.current_photo,

                text=""
            )


            # ================================================
            # Reset Results
            # ================================================

            self.prediction_label.configure(

                text="Prediction: ---"
            )

            self.confidence_label.configure(

                text="Confidence: ---"
            )


            self.selected_model_label.configure(

                text="Selected Model: ---"
            )


            self.status_label.configure(

                text=(
                    f"Status: Image Loaded - "
                    f"{Path(file_path).name}"
                )
            )


        except Exception as error:

            messagebox.showerror(

                "Error",

                f"Could not load image:\n\n{error}"
            )


    # ========================================================
    # Model Selection
    # ========================================================

    def on_model_selected(
        self,
        selected_model
    ):

        self.current_model = None

        self.status_label.configure(

            text=(
                f"Status: Selected Model - "
                f"{selected_model}"
            )
        )


    # ========================================================
    # Load Selected Model
    # ========================================================

    def get_model(
        self,
        model_name
    ):

        if model_name in self.loaded_models:

            return self.loaded_models[
                model_name
            ]


        # ====================================================
        # Baseline CNN
        # ====================================================

        if model_name == "Baseline CNN":

            model = BaselineCNN(

                num_classes=NUM_CLASSES
            )

            model_path = (
                BASELINE_MODEL_PATH
            )


        # ====================================================
        # ResNet18
        # ====================================================

        elif model_name == "ResNet18":

            model = ResNet18Model(

                num_classes=NUM_CLASSES,

                pretrained=False
            )

            model_path = (
                RESNET18_MODEL_PATH
            )


        # ====================================================
        # EfficientNet-B0
        # ====================================================

        elif model_name == "EfficientNet-B0":

            model = EfficientNetB0Model(

                num_classes=NUM_CLASSES,

                pretrained=False
            )

            model_path = (
                EFFICIENTNET_B0_MODEL_PATH
            )


        else:

            raise ValueError(
                "Unknown model selected."
            )


        # ====================================================
        # Load Checkpoint
        # ====================================================

        checkpoint = torch.load(

            model_path,

            map_location=DEVICE
        )


        if "model_state_dict" in checkpoint:

            model.load_state_dict(

                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )


        # ====================================================
        # Move Model To Device
        # ====================================================

        model = model.to(
            DEVICE
        )


        model.eval()


        # ====================================================
        # Cache Model
        # ====================================================

        self.loaded_models[
            model_name
        ] = model


        return model


    # ========================================================
    # Predict Image
    # ========================================================

    def predict_image(self):

        # ====================================================
        # Reset Stop Flag
        # ====================================================

        self.stop_requested = False


        # ====================================================
        # Check Image
        # ====================================================

        if self.current_image is None:

            messagebox.showwarning(

                "No Image",

                "Please load an MRI image first."
            )

            return


        # ====================================================
        # Get Selected Model
        # ====================================================

        model_name = (
            self.model_combobox.get()
        )


        if not model_name:

            messagebox.showwarning(

                "No Model",

                "Please select a model."
            )

            return


        try:

            self.status_label.configure(

                text=(
                    f"Status: Loading "
                    f"{model_name}..."
                )
            )

            self.update_idletasks()


            # =================================================
            # Load Model
            # =================================================

            model = self.get_model(
                model_name
            )


            # =================================================
            # Check Stop
            # =================================================

            if self.stop_requested:

                self.status_label.configure(

                    text="Status: Prediction Stopped"
                )

                return


            # =================================================
            # Prepare Image
            # =================================================

            image_tensor = self.transform(

                self.current_image
            )


            image_tensor = image_tensor.unsqueeze(
                0
            )


            image_tensor = image_tensor.to(
                DEVICE
            )


            # =================================================
            # Prediction
            # =================================================

            with torch.no_grad():

                outputs = model(
                    image_tensor
                )


                probabilities = torch.softmax(

                    outputs,

                    dim=1
                )


                confidence, prediction = torch.max(

                    probabilities,

                    dim=1
                )


            # =================================================
            # Check Stop
            # =================================================

            if self.stop_requested:

                self.status_label.configure(

                    text="Status: Prediction Stopped"
                )

                return


            # =================================================
            # Get Results
            # =================================================

            predicted_class_index = (

                prediction.item()
            )


            predicted_class = (

                CLASS_NAMES[
                    predicted_class_index
                ]
            )


            confidence_value = (

                confidence.item() * 100
            )


            # =================================================
            # Update Results
            # =================================================

            self.selected_model_label.configure(

                text=(
                    f"Selected Model: "
                    f"{model_name}"
                )
            )


            self.prediction_label.configure(

                text=(
                    f"Prediction: "
                    f"{predicted_class}"
                )
            )


            self.confidence_label.configure(

                text=(
                    f"Confidence: "
                    f"{confidence_value:.2f}%"
                )
            )


            self.status_label.configure(

                text=(
                    "Status: Prediction "
                    "Completed Successfully"
                )
            )


        except Exception as error:

            messagebox.showerror(

                "Prediction Error",

                f"An error occurred:\n\n{error}"
            )


            self.status_label.configure(

                text="Status: Error During Prediction"
            )


    # ========================================================
    # Load Best Model Information
    # ========================================================

    def load_best_model_info(self):

        try:

            csv_path = (

                EVALUATION_DIR

                / "model_comparison_results.csv"
            )


            if not csv_path.exists():

                self.best_model_label.configure(

                    text=(
                        "Best Model: "
                        "Comparison file not found"
                    )
                )

                return


            results_df = pd.read_csv(

                csv_path
            )


            if (

                "Model" not in results_df.columns

                or

                "F1-Score" not in results_df.columns

            ):

                self.best_model_label.configure(

                    text=(
                        "Best Model: "
                        "Invalid comparison file"
                    )
                )

                return


            # =================================================
            # Find Best Model
            # =================================================

            best_model_index = (

                results_df[
                    "F1-Score"
                ].idxmax()
            )


            best_model = (

                results_df.loc[

                    best_model_index,

                    "Model"
                ]
            )


            best_f1 = (

                results_df.loc[

                    best_model_index,

                    "F1-Score"
                ] * 100
            )


            best_accuracy = (

                results_df.loc[

                    best_model_index,

                    "Accuracy"
                ] * 100
            )


            # =================================================
            # Display Best Model
            # =================================================

            self.best_model_label.configure(

                text=(

                    f"Best Model: {best_model}\n"

                    f"Accuracy: "
                    f"{best_accuracy:.2f}%  |  "

                    f"F1-Score: "
                    f"{best_f1:.2f}%"
                )
            )


        except Exception as error:

            self.best_model_label.configure(

                text=(
                    "Best Model: "
                    "Unable to load results"
                )
            )

            print(
                "Best Model Error:",
                error
            )


    # ========================================================
    # Stop Prediction
    # ========================================================

    def stop_prediction(self):

        self.stop_requested = True

        self.status_label.configure(

            text="Status: Prediction Stopped"
        )


        self.prediction_label.configure(

            text="Prediction: Stopped"
        )


        self.confidence_label.configure(

            text="Confidence: ---"
        )


    # ========================================================
    # Exit Application
    # ========================================================

    def exit_application(self):

        result = messagebox.askyesno(

            "Exit",

            "Are you sure you want to exit?"
        )


        if result:

            self.destroy()


# ============================================================
# 6. Run Application
# ============================================================

if __name__ == "__main__":

    app = BrainTumorGUI()

    app.mainloop()