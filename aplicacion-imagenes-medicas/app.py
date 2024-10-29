from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import random
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, sobel
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# Configuraciones
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Verificar extensión de archivo
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verificar si se subió un archivo
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Obtener parámetros del formulario
            estilo = request.form.get('style', 'noni')
            tamanio_torneo = int(request.form.get('tour_size', 4))
            tamano_poblacion = int(request.form.get('tamano_poblacion', 100))
            generaciones = int(request.form.get('generaciones', 500))
            tasa_mutacion = float(request.form.get('tasa_mutacion', 0.01))
            num_padres = int(request.form.get('num_padres', 2))
            metodo_seleccion = request.form.get('metodo_seleccion', 'elite')
            metodo_cruce = request.form.get('metodo_cruce', 'uniforme')
            metodo_mutacion = request.form.get('metodo_mutacion', 'simple')
            kernel=(
                int(request.form.get('filas_kernel', '3')),
                int(request.form.get('cols_kernel', '3'))   
            )
            # Procesar la imagen
            imagen_mejorada = procesar_imagen(filepath, tamano_poblacion, generaciones, tasa_mutacion, num_padres, tamanio_torneo, estilo, metodo_seleccion, metodo_cruce, metodo_mutacion,kernel)

            # Guardar la imagen mejorada
            output_filename = 'mejorada_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            imagen_mejorada.save(output_path)
            print(filename)
            return render_template('index.html', original_image=filename, processed_image=output_filename)
    print("gg")
    return render_template('index.html')


def procesar_imagen(filepath, TAMANO_POBLACION, GENERACIONES, TASA_MUTACION, NUM_PADRES, TOUR_SIZE, STYLE, selection_method, crossover_method, mutation_method,kernel):
    radiografia = Image.open(filepath).convert("L")
    WIDTH, HEIGHT = radiografia.size
    radiografia = radiografia.resize((256, 256))
    radiografia_np = np.array(radiografia) / 255.0

    # Función para crear un filtro inicial
    def crear_individuo():
        return np.random.uniform(0, 1, kernel)

    # Función para aplicar un filtro a una imagen
    def aplicar_filtro(imagen, filtro):
        if STYLE == 'noni':
            return convolve(imagen, filtro)
        elif STYLE == 'sebas':
            imagen_filtrada = convolve(imagen, filtro)
            imagen_filtrada = (imagen_filtrada - imagen_filtrada.min()) / (imagen_filtrada.max() - imagen_filtrada.min())
            return imagen_filtrada

    def calcular_borde(imagen):
        grad_x = sobel(imagen, axis=0)
        grad_y = sobel(imagen, axis=1)
        return np.hypot(grad_x, grad_y)

    # Función de fitness con PSNR o SSIM
    def fitness(individuo):
        if STYLE == 'noni':
            imagen_filtrada = aplicar_filtro(radiografia_np, individuo)
            ssim_value = ssim(radiografia_np, imagen_filtrada, data_range=imagen_filtrada.max() - imagen_filtrada.min())
            bordes_original = calcular_borde(radiografia_np)
            bordes_filtrados = calcular_borde(imagen_filtrada)
            perdida_bordes = np.mean((bordes_original - bordes_filtrados) ** 2)
            return ssim_value - 0.5 * perdida_bordes
        elif STYLE == 'sebas':
            imagen_filtrada = aplicar_filtro(radiografia_np, individuo)
            mse = np.mean(abs(radiografia_np - imagen_filtrada))
            if mse == 0:
                psnr_value = float('inf')
            else:
                MAX = 1.0
                psnr_value = 10 * np.log10(MAX**2 / mse)
            return psnr_value

    # Métodos de selección
    def seleccion(poblacion, method='elite'):
        if method == 'elite':
            poblacion = sorted(poblacion, key=lambda x: fitness(x), reverse=True)
            return poblacion[:NUM_PADRES]
        elif method == 'ruleta':
            total_fitness = sum(fitness(ind) for ind in poblacion)
            selection_probs = [fitness(ind) / total_fitness for ind in poblacion]
            return random.choices(poblacion, weights=selection_probs, k=NUM_PADRES)
        elif method == 'aptitud':
            poblacion = sorted(poblacion, key=lambda x: fitness(x), reverse=True)
            top_half = poblacion[:len(poblacion) // 2]
            return random.sample(top_half, k=NUM_PADRES)
        elif method == 'muestreo_estocastico':
            total_fitness = sum(fitness(ind) for ind in poblacion)
            selection_probs = [fitness(ind) / total_fitness for ind in poblacion]
            distance = 1.0 / NUM_PADRES
            start_point = random.uniform(0, distance)
            points = [start_point + i * distance for i in range(NUM_PADRES)]
            selected = []
            cumulative_prob = 0
            j = 0
            for ind, prob in zip(poblacion, selection_probs):
                cumulative_prob += prob
                while j < NUM_PADRES and cumulative_prob > points[j]:
                    selected.append(ind)
                    j += 1
            return selected
        elif method == 'torneo':
            selected = []
            for _ in range(NUM_PADRES):
                tournament = random.sample(poblacion, TOUR_SIZE)
                winner = max(tournament, key=fitness)
                selected.append(winner)
            return selected
        elif method == 'rango':
            poblacion = sorted(poblacion, key=lambda ind: fitness(ind))
            rank_weights = [i + 1 for i in range(len(poblacion))]
            total_weight = sum(rank_weights)
            selection_probs = [weight / total_weight for weight in rank_weights]
            return random.choices(poblacion, weights=selection_probs, k=NUM_PADRES)

    # Función de cruce
    def cruce(padre1, padre2, method='uniforme'):
        size = padre1.shape
        hijo = np.copy(padre1)
        if method == 'uniforme':
            mask = np.random.randint(0, 2, size).astype(bool)
            hijo[mask] = padre2[mask]
        elif method == 'un_punto':
            punto = random.randint(1, size[0] - 1)
            hijo[:punto, :] = padre2[:punto, :]
        elif method == 'dos_puntos':
            punto1, punto2 = sorted(random.sample(range(size[0]), 2))
            hijo[punto1:punto2, :] = padre2[punto1:punto2, :]
        elif method == 'punto_medio':
            hijo[:size[0] // 2, :] = padre1[:size[0] // 2, :]
            hijo[size[0] // 2:, :] = padre2[size[0] // 2:, :]
        return hijo

    # Función de mutación
    def mutacion(individuo, method='simple'):
        mutante = np.copy(individuo)
        if method == 'simple':
            for i in range(mutante.shape[0]):
                for j in range(mutante.shape[1]):
                    if random.random() < TASA_MUTACION:
                        mutante[i, j] = np.random.uniform(0, 1)
        elif method == 'swap':
            indices = [(i, j) for i in range(mutante.shape[0]) for j in range(mutante.shape[1])]
            i1, i2 = random.sample(indices, 2)
            mutante[i1], mutante[i2] = mutante[i2], mutante[i1]
        elif method == 'crecimiento':
            for i in range(mutante.shape[0]):
                for j in range(mutante.shape[1]):
                    if random.random() < TASA_MUTACION:
                        mutante[i, j] *= np.random.uniform(1.0, 1.5)
                        mutante[i, j] = max(0, min(1, mutante[i, j]))
        elif method == 'reduccion':
            for i in range(mutante.shape[0]):
                for j in range(mutante.shape[1]):
                    if random.random() < TASA_MUTACION:
                        mutante[i, j] *= np.random.uniform(0.5, 1.0)
                        mutante[i, j] = max(0, mutante[i, j])
        return mutante

    # Algoritmo Genético Principal
    def algoritmo_genetico():
        poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]
        for generacion in range(GENERACIONES):
            padres = seleccion(poblacion, method=selection_method)
            nueva_poblacion = []
            while len(nueva_poblacion) < TAMANO_POBLACION:
                padre1, padre2 = random.sample(padres, 2)
                hijo = cruce(padre1, padre2, method=crossover_method)
                hijo_mutado = mutacion(hijo, method=mutation_method)
                nueva_poblacion.append(hijo_mutado)
            poblacion = nueva_poblacion

            if generacion % 50 == 0:
                mejor = max(poblacion, key=fitness)
                print(f"Generación {generacion}, Fitness: {fitness(mejor)}")

        mejor = max(poblacion, key=fitness)
        return mejor

    # Ejecutar el AG
    mejor_filtro = algoritmo_genetico()

    # Aplicar el mejor filtro a la imagen original
    imagen_final = aplicar_filtro(radiografia_np, mejor_filtro)
    imagen_final = (imagen_final * 255).astype(np.uint8)
    imagen_final_pil = Image.fromarray(imagen_final)
    imagen_final_pil = imagen_final_pil.resize((WIDTH, HEIGHT))
    return imagen_final_pil



if __name__ == '__main__':
    app.run(debug=True)
