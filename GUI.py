from tkinter import *
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlg
from tkinter import ttk


class GUI:
    """Class for whole GUI process"""

    def __init__(self):
        """Defining widgets"""
        self.root = Tk()
        self.root.geometry("600x600")
        self.root.title("Genetic algorithm")

        self.options = ['Eggholder function', 'Drop-wave function', 'Cross-in-tray function', 'Bukin Function N. 6',
                        'Ackley Function', 'Holder Table Function', 'Levy Function N. 13', 'Rastrigin Function']

        self.function_frame = LabelFrame(self.root, text="Function", width=260, height=190)
        self.define_label = Label(self.root, text='Choose function to optimize:', justify='left')
        self.enter_function = ttk.Combobox(self.root, values=self.options)

        self.min_x_label = Label(self.root, text='Minimum x:')
        self.min_x = DoubleVar()
        self.min_x_entry = Entry(self.root, width=35, textvariable=self.min_x)
        self.max_x_label = Label(self.root, text='Maximum x:')
        self.max_x = DoubleVar()
        self.max_x_entry = Entry(self.root, width=35, textvariable=self.max_x)

        self.state = True
        self.start_button = Button(self.root, text='Start', width=10, height=2, command=self.start_genetic)
        self.stop_button = Button(self.root, text='Stop', width=10, height=2, command=self.change_state)

        self.population_frame = LabelFrame(self.root, text="Population", width=200, height=150)
        self.population_label = Label(self.root, text='Population size')
        self.population_num = IntVar()
        self.population_number = Spinbox(self.root, from_=0, to=100, textvariable=self.population_num)

        self.precision_label = Label(self.root, text='Precision')
        self.precision = DoubleVar()
        self.precision_number = Entry(self.root, textvariable=self.precision)

        self.crossover_frame = LabelFrame(self.root, text="Crossover", width=200, height=100)
        self.crossover_label = Label(self.root, text="Crossover probability")
        self.crossover_probability = DoubleVar()
        self.crossover_number = Entry(self.root, textvariable=self.crossover_probability)

        self.mutation_frame = LabelFrame(self.root, text="Mutation", width=200, height=100)
        self.mutation_label = Label(self.root, text="Mutation probability")
        self.mutation_probability = DoubleVar()
        self.mutation_number = Entry(self.root, textvariable=self.mutation_probability)

        self.iteration_frame = LabelFrame(self.root, text="Iteration", width=200, height=100)
        self.iteration_label = Label(self.root, text="Iteration number")
        self.iteration = IntVar()
        self.iteration_number = Entry(self.root, textvariable=self.iteration)

        self.text_frame = LabelFrame(self.root, text="Result", width=200, height=100)
        self.text = StringVar("")
        self.text.set(f"Minimum equal:\nAt:\nIn generation number:")
        self.result = Label(self.root, textvariable=self.text, justify='left')

        self.gen_no = StringVar("")
        self.gen_no.set("Generation: 1")
        self.actual_gen = Label(self.root, textvariable=self.gen_no, justify='left')

        self.fig = plt.figure(figsize=(3, 3))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.options_map = {
            self.options[0]: [
                "-(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))",
                -512, 512],
            self.options[1]: ["-((1 + np.cos(12 * np.sqrt(x1 * x1 + x2 * x2))) / (0.5 * (x1 * x1 + x2 * x2) + 2))",
                              -5.12,
                              5.12],
            self.options[2]: [
                "-0.0001 * np.power((np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1 * x1 + x2 * x2) / np.pi))) + 1), 0.1)",
                -10, 10],
            self.options[3]: ["100 * np.sqrt(np.abs(x2 - 0.01 * x1 * x1)) + 0.01*np.abs(x1 + 10)", -10, 10],
            self.options[4]: [
                "-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 * x1 + x2 * x2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)",
                -32.768, 32.768],
            self.options[5]: [
                "-np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1 * x1 + x2 * x2) / np.pi)))",
                -10, 10],
            self.options[6]: [
                "np.power(np.sin(3 * np.pi * x1), 2) + np.power(x1 - 1, 2) * (1 + np.power(3 * np.pi * x2, 2)) + np.power(x2 - 1, 2) * (1 + np.power(np.sin(2 * np.pi * x2), 2))",
                -10, 10],
            self.options[7]: ["20 + x1 * x1 - 10*np.cos(2 * np.pi * x1) + x2 * x2 -10 * np.cos(2 * np.pi * x2)", -5.12,
                              5.12],
        }
        self.set_widgets()
        self.change_details()

        self.root.mainloop()

    def set_widgets(self):
        """Details of widgets - placing, fonts setting etc."""
        self.function_frame.place(x=0, y=0)
        self.define_label.config(font=("Arial", 12))
        self.define_label.place(x=20, y=20)

        self.enter_function.current(0)
        self.enter_function.config(font=("Arial", 10))
        self.enter_function.place(x=20, y=45)
        self.enter_function.bind('<<ComboboxSelected>>', self.change_details)

        self.min_x_label.config(font=("Arial", 10))
        self.min_x_label.place(x=20, y=80)

        self.min_x.set(self.options_map[self.options[self.enter_function.current()]][1])
        self.min_x_entry.place(x=20, y=100)

        self.max_x_label.config(font=("Arial", 10))
        self.max_x_label.place(x=20, y=130)

        self.max_x.set(self.options_map[self.options[self.enter_function.current()]][2])
        self.max_x_entry.place(x=20, y=150)

        self.start_button.place(x=20, y=200)
        self.stop_button.place(x=110, y=200)

        self.population_frame.place(x=340, y=0)

        self.population_label.config(font=("Arial", 10))
        self.population_label.place(x=360, y=20)

        self.population_num.set(40)
        self.population_number.place(x=363, y=45)

        self.precision_label.config(font=("Arial", 10))
        self.precision_label.place(x=360, y=80)

        self.precision.set(1e-4)
        self.precision_number.place(x=363, y=100)

        self.crossover_frame.place(x=340, y=150)
        self.crossover_label.config(font=("Arial", 10))
        self.crossover_label.place(x=360, y=170)
        self.crossover_probability.set(0.7)
        self.crossover_number.place(x=363, y=200)

        self.mutation_frame.place(x=340, y=250)
        self.mutation_label.config(font=("Arial", 10))
        self.mutation_label.place(x=360, y=270)
        self.mutation_probability.set(0.01)
        self.mutation_number.place(x=363, y=300)

        self.iteration_frame.place(x=340, y=350)
        self.iteration_label.config(font=("Arial", 10))
        self.iteration_label.place(x=360, y=370)
        self.iteration.set(100)
        self.iteration_number.place(x=363, y=400)

        self.text_frame.place(x=340, y=450)
        self.result.config(font=("Arial", 12))
        self.result.place(x=360, y=470)

        self.actual_gen.config(font=("Arial", 12))
        self.actual_gen.place(x=20, y=550)

    def start_genetic(self):
        """Main loop of genetic algorithm"""
        self.state = True
        gen = GeneticAlg(2, self.population_num.get(), self.precision_number.get(),
                         self.crossover_probability.get(), self.mutation_probability.get(),
                         self.min_x.get(), self.max_x.get(), self.iteration.get())
        graph_points = []
        gen.finish_point, gen.dx = gen.nbits()
        pop = gen.gen_population()
        oc = gen.evaluate_population(gen.obj_func, pop,
                                     self.options_map[self.options[self.enter_function.current()]][0])
        best_a, best_val = gen.get_best(pop, oc)
        best_arg = gen.decode_individual(best_a)

        list_best = [best_val]
        list_mean = [np.average(oc)]
        list_best_generation = [best_val]
        best_sol = 0

        pop_d = []
        for os in pop:
            pop_d.append(gen.decode_individual(os))
        pop_d = np.array(pop_d)
        graph_points.append(pop_d)

        for i in range(gen.max_iter):
            if self.state is False:
                break
            pop = gen.roulette(pop, oc)
            pop = gen.cross(pop, gen.cross_probability)
            pop = gen.mutate(pop, gen.mutation_probability)
            oc = gen.evaluate_population(gen.obj_func, pop,
                                         self.options_map[self.options[self.enter_function.current()]][0])
            cor = np.apply_along_axis(self.decode_local, 1, pop, gen)
            x = self.plot_f(cor, oc)
            for point in x:
                point.remove()
            val = np.min(oc)
            list_best_generation.append(val)
            if val < best_val:
                best_val = val
                best_sol = i
                best_arg = gen.decode_individual(pop[np.argmin(oc)])
                self.text.set(
                    f"Minimum equal: {round(val, 2)}\nAt: {np.around(best_arg, 2)}\nIn generation number: {i + 1}")
            list_best.append(best_val)
            list_mean.append(np.average(oc))
            if i == gen.max_iter / 2 or i == gen.max_iter - 1:
                pop_d = []
                for os in pop:
                    pop_d.append(gen.decode_individual(os))
                pop_d = np.array(pop_d)
                graph_points.append(pop_d)
            self.gen_no.set(f"Generation: {i + 1} / {gen.max_iter}")
        return best_val, best_sol, best_arg, list_best, list_best_generation, list_mean, graph_points

    def plot_f(self, x_values, fx_values):
        """Plotting charts"""
        appended = []
        for i, j in enumerate(fx_values):
            appended.append(self.ax.scatter(x_values[i][0], x_values[i][1], j, color='red', s=6))
        self.fig.canvas.draw()
        self.root.update()
        return appended

    @staticmethod
    def decode_local(pop, gen):
        return GeneticAlg.decode_individual(gen, pop)

    def change_details(self, *args):
        """Switching between functions"""
        actual_fun = self.options_map[self.options[self.enter_function.current()]]
        self.min_x.set(actual_fun[1])
        self.max_x.set(actual_fun[2])

        plt.ion()
        self.ax.clear()

        points = np.linspace(self.options_map[self.options[self.enter_function.current()]][1],
                             self.options_map[self.options[self.enter_function.current()]][2], 50)
        x, y = np.meshgrid(points, points)
        z = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(len(points)):
                z[i, j] = GeneticAlg.obj_func((x[i, j], y[i, j]), actual_fun[0])

        self.ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                             linewidth=0.3, antialiased=True, alpha=0.5)
        self.ax.view_init(30, 200)

        canvas_ = FigureCanvasTkAgg(self.fig, self.root)
        canvas_.get_tk_widget().place(x=20, y=250)

    def change_state(self):
        """Changing state for Start/Stop operations"""
        self.state = False


if __name__ == '__main__':
    gui = GUI()
