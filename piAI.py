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
    x = np.random.randint(1, 20)  # 1 és 19 között
    y = np.random.randint(1, 20)
    op = np.random.choice(['+', '-', '*', '/'])
    # Biztonsági ellenőrzés nullával való osztásra
    if op == '/' and y == 0:
        y = 1
    return x, y, op

def encode_input(x, y, op):
    ops = ['+', '-', '*', '/']
    op_index = ops.index(op)
    return np.array([[x, y, op_index]])

def correct_result(x, y, op):
    try:
        if op == '+':
            return x + y
        elif op == '-':
            return x - y
        elif op == '*':
            return x * y
        elif op == '/':
            return x / y
    except ZeroDivisionError:
        return 0

def compress_output(val, x, y, op):
    """
    Kompresszor, amely korlátozza az AI által adott értéket,
    hogy ne legyen irreálisan nagy.
    Például 1+1 nem lehet több 10-nél.
    Itt rugalmas szabályokat alkalmazunk.
    """
    # Határérték alapértelmezett (pl. max 10-szeres)
    max_val = 10 * max(x, y)
    
    # Minimum érték, ha művelet negatív lehet (kivonás)
    min_val = -max_val
    
    # Speciális szabályok opciók szerint
    if op == '+':
        max_val = 10
        min_val = 0
    elif op == '-':
        max_val = 10
        min_val = -10
    elif op == '*':
        max_val = 400  # 20*20 = 400
        min_val = 0
    elif op == '/':
        max_val = 20  # osztás eredménye nem lesz nagyobb mint max operandus
        min_val = 0
    
    # Kompresszor alkalmazása (kicsinyítés)
    if val > max_val:
        val = max_val
    elif val < min_val:
        val = min_val
    return val

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
        # Alap tanítás minimális adatokkal, hogy legyen valid input méret
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
            self.load_step = len(self.errors) if self.errors else 0
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

            # Kompresszor alkalmazása
            ai_guess = compress_output(ai_guess, x, y, op)

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

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.signal import argrelextrema
        from scipy.optimize import curve_fit

        errors_array = np.array(self.errors)
        steps = np.arange(len(errors_array))

        # Mozgóátlagok
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')

        ma10 = moving_average(errors_array, 10)
        ma50 = moving_average(errors_array, 50)
        ma200 = moving_average(errors_array, 200) if len(errors_array) >= 200 else None

        # Szórásgörbe (±1 std)
        std_dev = np.std(errors_array)
        avg_error = np.mean(errors_array)
        upper_std = avg_error + std_dev
        lower_std = avg_error - std_dev

        # Exponenciális illesztés
        def exp_func(x, a, b, c): return a * np.exp(-b * x) + c
        try:
            popt, _ = curve_fit(exp_func, steps, errors_array, maxfev=10000)
            exp_fit = exp_func(steps, *popt)
        except Exception:
            exp_fit = None

        # Derivált kiszámítása
        error_diff = np.diff(errors_array, prepend=errors_array[0])

        # Lokális minimumok (javulási pontok)
        local_minima = argrelextrema(errors_array, np.less)[0]

        # Ábra létrehozása
        plt.figure(figsize=(14, 8))
        plt.plot(steps, errors_array, color='blue', alpha=0.6, label='Hiba')
        plt.plot(steps[9:], ma10, color='orange', label='Mozgóátlag (10)')
        plt.plot(steps[49:], ma50, color='green', label='Mozgóátlag (50)')
        if ma200 is not None:
            plt.plot(steps[199:], ma200, color='red', label='Mozgóátlag (200)')
        if exp_fit is not None:
            plt.plot(steps, exp_fit, '--', color='cyan', label='Várható csökkenés (exp)')

        # Szórás sáv
        plt.fill_between(steps, lower_std, upper_std, color='gray', alpha=0.1, label='±1 szórás')

        # Medián + átlag
        median_error = np.median(errors_array)
        plt.axhline(median_error, color='purple', linestyle=':', label=f'Medián: {median_error:.3f}')
        plt.axhline(avg_error, color='black', linestyle='--', label=f'Átlag: {avg_error:.3f}')

        # Derivált
        plt.plot(steps, error_diff, color='gray', linestyle='--', alpha=0.4, label='Hiba deriváltja')

        # Modell betöltési pont
        if hasattr(self, 'load_step') and self.load_step is not None:
            plt.axvline(self.load_step, color='red', linestyle='--', label=f'Modell betöltve: {self.load_step}')

        # Lokális minimumok
        plt.scatter(local_minima, errors_array[local_minima], marker='o', color='lime', label='Javulási pontok')

        plt.xlabel("Iteráció")
        plt.ylabel("Abszolút hiba")
        plt.title("Tanulási hiba elemzése (részletes)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # + Hiba hisztogram külön ábrában
        plt.figure(figsize=(8, 5))
        plt.hist(errors_array, bins=40, color='teal', edgecolor='black')
        plt.title("Hiba eloszlás (hisztogram)")
        plt.xlabel("Hibaérték")
        plt.ylabel("Előfordulás")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # + Log skálás hibaábra
        plt.figure(figsize=(12, 6))
        plt.semilogy(steps, errors_array, label="Hibák (log skála)", color='darkblue')
        plt.title("Tanulási hiba (logaritmikus ábrázolás)")
        plt.xlabel("Iteráció")
        plt.ylabel("Log(hiba)")
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()


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
