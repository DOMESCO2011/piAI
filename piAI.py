import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib
import os
import threading

# --- Segédfüggvények ---

def generate_random_example():
    x = np.random.randint(1, 20)
    y = np.random.randint(1, 20)
    op = np.random.choice(['+', '-', '*', '/'])
    if op == '/' and y == 0:
        y = 1
    return x, y, op

def encode_input(x, y, op):
    ops = ['+', '-', '*', '/']
    op_index = ops.index(op)
    return np.array([[x, y, op_index]])

def correct_result(x, y, op):
    try:
        if op == '+': return x + y
        elif op == '-': return x - y
        elif op == '*': return x * y
        elif op == '/': return x / y
    except ZeroDivisionError:
        return 0

# --- Tanító osztály ---

class AITrainer:
    def __init__(self, log_callback):
        self.model = None
        self.errors = []
        self.noOfEquals = 0
        self.noOfErrors = 0
        self.log_callback = log_callback
        self.model_file = "ai_model.joblib"

    def init_model(self):
        self.model = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1, warm_start=True)
        X_train = np.array([[1,1,0],[2,2,0],[3,3,0],[4,4,0],[5,5,0]])
        y_train = np.array([2,4,6,8,10])
        self.model.fit(X_train, y_train)

    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_file)
            self.log_callback("💾 Modell elmentve!")

    def load_model(self):
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            self.log_callback("📂 Modell betöltve!")
        else:
            self.log_callback("❗ Nincs elmentett modell!")

    def auto_teach(self, max_steps, target_streak):
        if self.model is None:
            self.init_model()

        self.errors = []
        self.noOfEquals = 0
        self.noOfErrors = 0
        streak = 0
        step = 0

        while step < max_steps and streak < target_streak:
            step += 1
            x, y, op = generate_random_example()
            input_encoded = encode_input(x, y, op)
            real_result = correct_result(x, y, op)

            try:
                ai_guess = self.model.predict(input_encoded)[0]
            except:
                ai_guess = 0

            error = abs(ai_guess - real_result)
            self.errors.append(error)

            if error < 0.1:
                streak += 1
                feedback = "✔️"
                self.noOfEquals += 1
            else:
                streak = 0
                feedback = "❌"
                self.model.partial_fit(input_encoded, [real_result])
                self.noOfErrors += 1

            self.log_callback(f"{step:04d}. {x} {op} {y} = {real_result} | AI: {ai_guess:.4f} | {feedback} | Streak: {streak}")

        self.log_callback("✅ Tanítás befejezve!")

    def plot_errors(self):
        if not self.errors:
            messagebox.showinfo("Info", "Nincs hiba adat.")
            return
        plt.figure(figsize=(10,5))
        plt.plot(self.errors)
        plt.xlabel("Iterációk")
        plt.ylabel("Abszolút hiba")
        plt.title("Tanulási hiba")
        plt.grid(True)
        plt.show()

        avg_error = np.mean(self.errors)
        min_error = np.min(self.errors)
        max_error = np.max(self.errors)
        all_corr = self.noOfEquals
        all_error = self.noOfErrors
        szazalekkorrekt = 100 * all_corr / all_error

        stats = (f"Átlagos hiba: {avg_error:.4f}\n"
                 f"Minimális hiba: {min_error:.4f}\n"
                 f"Maximális hiba: {max_error:.4f}\n"
                 f"Találatok száma: {self.noOfEquals}\n"
                 f"Hibák száma: {self.noOfErrors}\n"
                 f"Találatok százaléka: {szazalekkorrekt}")

        messagebox.showinfo("Statisztika", stats)

# --- Tkinter GUI ---

class TrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Matematikai Művelet Tanító")

        self.trainer = AITrainer(self.log)

        self.control_frame = ttk.Frame(root, padding=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(self.control_frame, text="Max. iteráció:").pack(side=tk.LEFT)
        self.iter_entry = ttk.Entry(self.control_frame, width=10)
        self.iter_entry.insert(0, "10000")
        self.iter_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.control_frame, text="Szükséges helyesek egymás után:").pack(side=tk.LEFT)
        self.streak_entry = ttk.Entry(self.control_frame, width=5)
        self.streak_entry.insert(0, "10")
        self.streak_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.control_frame, text="Start tanítás", command=self.run_training).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.control_frame, text="Mentés", command=self.trainer.save_model).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Betöltés", command=self.trainer.load_model).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Grafikon & stat", command=self.trainer.plot_errors).pack(side=tk.LEFT, padx=10)

        self.log_text = tk.Text(root, height=20, bg="black", fg="lime", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def run_training(self):
        try:
            max_steps = int(self.iter_entry.get())
            streak = int(self.streak_entry.get())
        except ValueError:
            messagebox.showerror("Hiba", "Kérlek számokat adj meg!")
            return

        threading.Thread(target=self.trainer.auto_teach, args=(max_steps, streak), daemon=True).start()

# --- Futtatás ---

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainerApp(root)
    root.mainloop()
