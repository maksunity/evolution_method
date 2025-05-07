import neat
import numpy as np
import os
import pickle
import math
from torchvision import datasets
from PIL import Image, ImageDraw
from neat.parallel import ParallelEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import csv

warnings.filterwarnings("ignore", category=UserWarning)

# --- Параметры ---
CONFIG_PATH = 'config-feedforward.txt'
IMAGE_SIZE = (16, 16)  # размер входа
NUM_INPUTS = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_GENERATIONS = 50
OUTPUT_DIR = 'neat_output_mnist_parallel'
LOG_CSV = os.path.join(OUTPUT_DIR, 'training_log.csv')  # лог CSV
CHECKPOINT_EVERY = 5  # чекпоинты каждые N поколений
SUBSET_FRAC = 0.5  # константная подвыборка из общего датасета

class MNISTDataHolder:
    def __init__(self):
        self.train_images, self.train_labels = self._load_data(train=True)
        self.test_images, self.test_labels = self._load_data(train=False)

    def _load_data(self, train=True):
        dataset = datasets.MNIST(
            root='./mnist_data', train=train, download=True, transform=None
        )
        images, labels = [], []
        for img, label in dataset:
            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_np = np.array(img_resized) / 255.0  # нормализация
            images.append(img_np.flatten())
            labels.append(label)
        return np.array(images), np.array(labels)

# Инициализация данных
data_holder = MNISTDataHolder()


# Оценка генома по случайной подвыборке SUBSET_FRAC
def eval_single_genome_mnist(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    try:
        total = len(data_holder.train_images)
        subset_size = int(total * SUBSET_FRAC)
        indices = np.random.choice(total, subset_size, replace=False)
        correct = sum(
            np.argmax(net.activate(data_holder.train_images[i])) == data_holder.train_labels[i]
            for i in indices
        )
        return correct / subset_size
    except Exception as e:
        print(f"Ошибка оценки генома: {e}")
        return 0.0


# Функция запуска NEAT с ParallelEvaluator, чекпоинтами и логированием
def run_neat_mnist_parallel(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    if config.genome_config.num_inputs != NUM_INPUTS:
        raise ValueError(f"Несоответствие входов: {config.genome_config.num_inputs} vs {NUM_INPUTS}")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=CHECKPOINT_EVERY,
                                     filename_prefix=os.path.join(OUTPUT_DIR, 'chkpt-')))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_file = open(LOG_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['generation', 'best_fitness', 'avg_fitness'])

    num_workers = min(os.cpu_count(), 6)
    pe = ParallelEvaluator(num_workers, eval_single_genome_mnist)

    gen_counter = {'gen': 0}

    def fitness_wrapper(genomes, config_inner):
        gen = gen_counter['gen']
        # Распараллеливание оценки
        result = pe.evaluate(genomes, config_inner)

        fits = [g.fitness for _, g in genomes]
        csv_writer.writerow([gen, max(fits), float(np.mean(fits))])
        csv_file.flush()

        gen_counter['gen'] += 1
        return result

    winner = p.run(fitness_wrapper, NUM_GENERATIONS)
    csv_file.close()
    with open(os.path.join(OUTPUT_DIR, 'winner_genome.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    return winner, config, stats


# Постобработка: графики и анализ
def plot_training_curves():
    data = np.loadtxt(LOG_CSV, delimiter=',', skiprows=1)
    gens, best, avg = data[:,0], data[:,1], data[:,2]
    plt.figure()
    plt.plot(gens, best, label='Best')
    plt.plot(gens, avg, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Training Curves')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()
def post_training_analysis(net, data_holder):
    print("\n--- Анализ после обучения на тестовых данных ---")
    try:
        print("Получение предсказаний на тестовом наборе...")
        test_inputs = data_holder.test_images
        test_labels = data_holder.test_labels
        # Используем list comprehension для предсказаний
        preds = [np.argmax(net.activate(img)) for img in test_inputs]
        print(f"Предсказания получены для {len(preds)} тестовых образцов.")

        # 1. Расчет и вывод точности (Accuracy)
        accuracy = accuracy_score(test_labels, preds)
        print(f"Точность на тестовом наборе: {accuracy:.4f} ({int(accuracy * len(test_labels))}/{len(test_labels)})")

        # 2. Матрица ошибок (Confusion Matrix)
        print("Построение матрицы ошибок...")
        cm = confusion_matrix(test_labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()
        print("Матрица ошибок сохранена.")

        # 3. ROC-кривая для цифры 0 (опционально, можно для любой)
        print("Построение ROC-кривой для цифры 0...")
        raw_outputs = np.array([net.activate(img) for img in test_inputs])
        y_true_roc = (test_labels == 0).astype(int) # 1 если цифра 0, иначе 0
        # Используем выход нейрона для цифры 0 как оценку уверенности
        y_score_roc = raw_outputs[:, 0]
        fpr, tpr, _ = roc_curve(y_true_roc, y_score_roc)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Линия случайного угадывания
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Digit 0 (Test Set)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_digit0.png'))
        plt.close()
        print("ROC-кривая сохранена.")

        # 4. t-SNE визуализация выходов сети
        print("Построение t-SNE визуализации (может занять время)...")
        # Используем сырые выходы сети как признаки для t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30.0)
        features = tsne.fit_transform(raw_outputs)
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(features[:, 0], features[:, 1], c=test_labels, cmap='viridis', s=10)
        plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Digits")
        plt.title('t-SNE Visualization of Network Outputs (Test Set)')
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_outputs.png'))
        plt.close()
        print("t-SNE визуализация сохранена.")

    except Exception as e:
        print(f"Ошибка во время анализа после обучения: {e}")
        import traceback
        print(traceback.format_exc())

def display_sample_predictions(net, data_holder, total_images=10, images_per_plot=5):
    print(f"\n--- Отображение {total_images} случайных предсказаний ({images_per_plot} на файл) ---")
    try:
        test_images = data_holder.test_images
        test_labels = data_holder.test_labels
        num_available = len(test_images)

        if num_available == 0:
            print("Тестовый набор пуст, невозможно отобразить примеры.")
            return
        if total_images <= 0 or images_per_plot <= 0:
             print("Количество изображений (total_images и images_per_plot) должно быть положительным.")
             return

        actual_total_images = min(total_images, num_available)
        if actual_total_images < total_images:
             print(f"Внимание: В тестовом наборе доступно только {actual_total_images} изображений.")

        all_indices = np.random.choice(num_available, actual_total_images, replace=False)
        num_plots = math.ceil(actual_total_images / images_per_plot)

        indices_offset = 0
        for plot_idx in range(num_plots):
            start_idx = indices_offset
            end_idx = min(indices_offset + images_per_plot, actual_total_images)
            current_indices = all_indices[start_idx:end_idx]
            num_in_this_plot = len(current_indices)

            if num_in_this_plot == 0:
                continue

            fig, axes = plt.subplots(1, num_in_this_plot, figsize=(num_in_this_plot * 3, 4))
            if num_in_this_plot == 1:
                 axes = [axes]

            fig.suptitle(f'Примеры предсказаний (Часть {plot_idx + 1}/{num_plots})')

            for i, ax in enumerate(axes):
                idx = current_indices[i]
                flat_image = test_images[idx]
                true_label = test_labels[idx]
                image_2d = flat_image.reshape(IMAGE_SIZE)
                output = net.activate(flat_image)
                prediction = np.argmax(output)

                ax.imshow(image_2d, cmap='gray')
                ax.set_title(f"Предск: {prediction}\nИстина: {true_label}",
                             color=("green" if prediction == true_label else "red"))
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            filename = os.path.join(OUTPUT_DIR, f'sample_predictions_{plot_idx + 1}.png')
            plt.savefig(filename)
            print(f"График сохранен в {filename}")
            plt.close(fig)
            indices_offset += num_in_this_plot

    except Exception as e:
        print(f"Ошибка при отображении примеров предсказаний: {e}")
        import traceback
        print(traceback.format_exc())

# GUI: рисование и распознавание
class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.threshold = 0.3
        self.root = tk.Tk()
        self.root.title("Digit Recognizer with Controls")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.label = tk.Label(self.root, text="Нарисуйте цифру", font=('Arial', 14))
        self.label.pack()

        self.result_label = tk.Label(self.root, text="", font=('Arial', 16))
        self.result_label.pack()

        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text='Threshold').pack(side=tk.LEFT)
        self.thresh_slider = Scale(frame, from_=0, to=1, resolution=0.01,
                                   orient=HORIZONTAL, command=self.update_threshold)
        self.thresh_slider.set(self.threshold)
        self.thresh_slider.pack(side=tk.LEFT)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Распознать", command=self.recognize).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Очистить", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def update_threshold(self, val):
        self.threshold = float(val)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_point:
            self.canvas.create_line(self.last_point[0], self.last_point[1], x, y,
                                   width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_point, (x, y)], fill=0, width=15)
        self.last_point = (x, y)

    def reset(self, event):
        self.last_point = None
    def recognize(self):
        try:
            img = self.image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0

            # Инверсия (Белый фон -> 0, Черная цифра -> 1)
            img_array_inverted = 1.0 - img_array

            inputs = img_array_inverted.flatten()
            output = self.net.activate(inputs)

            output_stable = output - np.max(output)
            exp_outputs = np.exp(output_stable)
            probs = exp_outputs / np.sum(exp_outputs)
            pred = np.argmax(probs)
            conf = probs[pred]

            color = 'green' if conf >= self.threshold else 'red'
            self.result_label.config(text=f"Цифра: {pred} Уверенность: {conf:.1%}", fg=color)

            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Canvas) and hasattr(widget, 'figure'):
                     widget.destroy()
                elif isinstance(widget, FigureCanvasTkAgg):
                     widget.get_tk_widget().destroy()


            plt.switch_backend('Agg')
            fig_probs = plt.figure(figsize=(5, 2))
            plt.bar(range(10), probs)
            plt.ylim(0, 1)
            plt.title('Probabilities')
            plt.xticks(range(10))
            plt.tight_layout()

            canvas_probs = FigureCanvasTkAgg(fig_probs, master=self.root)
            canvas_probs.draw()
            canvas_probs.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {e}", fg='red')
            import traceback
            print(f"Ошибка в recognize: {traceback.format_exc()}")

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.result_label.config(text="")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, 'winner_genome.pkl')
    if os.path.exists(model_path):
        print(f"Загрузка обученного генома из {model_path}")
        with open(model_path, 'rb') as f:
            winner = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             CONFIG_PATH)
    else:
        winner, config, stats = run_neat_mnist_parallel(CONFIG_PATH)
    print("Создание сети из лучшего генома...")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    print("\nТестовая точность:")
    test_acc = sum(np.argmax(net.activate(img)) == label
                   for img, label in zip(data_holder.test_images, data_holder.test_labels)) / len(
        data_holder.test_images)
    print(f"Accuracy: {test_acc:.2%}")

    plot_training_curves()
    post_training_analysis(net, data_holder)
    display_sample_predictions(net, data_holder, total_images=100, images_per_plot=10)

    print("Запуск GUI для рисования...")
    drawer = DigitDrawer(net)
    drawer.run()
