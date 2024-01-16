"""
Proyecto Final de Sistemas Multiagentes
Jose Ignacio Alba Rodríguez
MUIA 2023/2024

Discrete Cuckoo Search vs Ant Colony System para problemas de TSPLIB
"""

import numpy as np
import random, math
import matplotlib.pyplot as plt
import time


"""
Parte 1: Definición del problema
    Estructura de Grafo
    Parser para obtener los problemas de TSPLIB
    
"""

class Nodo():
    def __init__(self, n_id, x, y):
        self.id = n_id
        self.pos = np.array( (x,y) )
        self.adyacentes = {}
    
    def __repr__(self):
        return f"{self.pos}"
    
            
class Grafo():
    def __init__(self, nodos, nombre, dim):
        self.dim = dim
        self.nombre = nombre
        self.nodos = {n.id : n for n in nodos}  
        self.aristas = {}
        self.simetrico = True
        for i, n1 in enumerate(nodos):
            for n2 in nodos[i+1:]:
                d = round(np.linalg.norm(n2.pos - n1.pos))
                n1.adyacentes[n2.id] = d
                n2.adyacentes[n1.id] = d
                self.aristas[(n1.id, n2.id)] = d
                self.aristas[(n2.id, n1.id)] = d
    
    # Función Objetivo
    def evalCamino(self, camino):
        suma = 0
        for i in range(len(camino)-1):
            a = camino[i]
            b = camino[i+1]
            suma += self.aristas[(a,b)]
        return suma
    
    # Representar los nodos y un camino entre ellos
    def plot(self, camino = None):
        posiciones = np.array( [nodo.pos for nodo in self.nodos.values()] )
        plt.scatter(posiciones[:,0], posiciones[:,1])
        if camino != None:
            camino = np.array([self.nodos[n_id].pos for n_id in camino])
            plt.plot(camino[:,0],camino[:,1])
    
    
def leerTSP(file):
    nodos = []
    with open(file, "r") as f:
        linea = f.readline().strip()
        while linea != 'NODE_COORD_SECTION':
            if linea[:4] == "NAME":
                name = linea[6:].strip()
            elif linea[:9] == "DIMENSION":
                dim = int(linea[11:])
            elif linea[:4] == "EDGE" and linea[-6:] != "EUC_2D":
                print(linea)
                print(linea[-6:])
                raise Exception("La distancia no es euclidea")
            linea = f.readline().strip()
            
        linea = f.readline().strip()            
        while linea != 'EOF':
            x = linea.split(" ")
            x = [numero for numero in x if numero != ""]
            n = Nodo(int(x[0]), float(x[1]), float(x[2]) )
            nodos.append(n)
            linea = f.readline().strip()
    return Grafo(nodos, name, dim)



"""
Parte 2: Cuckoo Search
    Distribución de Levy
    Movimientos locales en las soluciones del grafo
    Improved Cuckoo Search (ICS)
    Complete Cuckoo Search(CCS)

"""

# Movimientos Locales
def permutarConsecutivos(camino):
    l = camino[:]
    n = len(camino)-1
    i = np.random.choice(n)
    l[i], l[i+1] = l[i+1], l[i]
    if i == 0: l[-1] = l[0]
    elif i == n-1: l[0] = l[-1]
    return l
    
def two_opt(camino):
    n = len(camino)-1
    opciones = list(range(n))
    i = np.random.choice(opciones)
    opciones = [x for x in opciones if x not in {(i-1)%n, i, (i+1)%n}]  
    j = np.random.choice(opciones)
    if i < j: return [camino[i]] + camino[j:i:-1] + camino[j+1:-1] + camino[0:i+1]
    else: return     (camino[i:j:-1] + camino[i+1:-1] + camino[0:j+1] + [camino[i]])[::-1]
    
def double_bridge(camino):
    n = len(camino)-1
    puntos = []
    opciones = list(range(n))
    for i in range(4):    
        puntos.append(np.random.choice(opciones))
        opciones = [x for x in opciones if x not in {(puntos[-1]-1)%n, puntos[-1], (puntos[-1]+1)%n}]
    a, b, c, d = sorted(puntos)
    return [camino[a]] + camino[c+1:d+1] + camino[b+1:c+1] + camino[a+1:b+1] + camino[d+1:-1] + camino[:a+1]
  

# Levy Flights
def Levy(alpha = 1, beta = 1.5):
    num = math.gamma(1+beta) * math.sin(math.pi * beta /2)
    den = math.gamma((1+beta)/2) * beta * (2**((beta+1)/2))
    u = np.random.normal(0, (num/den)**(1/beta))
    v = np.random.normal(0, 1)
    return alpha*u/(np.abs(v)**(1/beta))

def CuckooMove(camino, alpha = 1, beta = 1.5, s0 = 1):
    s = Levy(alpha, beta)
    if s <= s0: return two_opt(camino)
    else: return double_bridge(camino)
    

# Improved Cuckoo Search
def CuckooSearchMejorado(g, n, it, pa, pc, s0=1, alpha = 1, beta = 1.5, init = True):
    # Inicialización
    # Si init es True, entonces se inicializa desde Nearest Neighbour. En caso contrario, se inicializa desde una solución aleatoria
    nidos, valores = [], []
    mejor_val, mejor_sol = float('inf'), []
    m = int(pc*n)
    for i in range(n):
        if init:
            # Inicialización desde Nearest Neighbour
            val, cam = NN_heuristic(g)
            nidos.append(cam)
            valores.append(val)
        else:
            # Inicialización aleatoria
            nodos = list(g.nodos.keys())
            x1 = random.sample(nodos, len(nodos))
            x1.append(x1[0])
            nidos.append(x1)
            valores.append(g.evalCamino(x1))
    
    # Cuerpo del algoritmo
    for _ in range(it):
        # Una fraccion pc de "cucos inteligentes" "empieza a buscar" (realiza un Levy flight)
        cuckoos_inteligentes = random.sample(range(n),m)
        for cuckoo in cuckoos_inteligentes:
            nuevo_camino = CuckooMove(nidos[cuckoo], alpha, beta, s0)
            val = g.evalCamino(nuevo_camino)
            if val < valores[cuckoo]:
                nidos[cuckoo] = nuevo_camino
                valores[cuckoo] = val
        
        # Un único cuco se desplaza a un nuevo nido
        cuckoo = random.randint(0,n-1)
        nuevo_camino = CuckooMove(nidos[cuckoo], alpha, beta, s0)
        val = g.evalCamino(nuevo_camino)
        nido = random.randint(0,n-1)
        if val < valores[nido]:
            nidos[nido] = nuevo_camino
            valores[nido] = val
        
        # Una fraccion de los peores cucos es abandonada y se generan nuevos
        ranking = sorted( list(range(n)), key = lambda i: -valores[i])  # Indices de peor a mejor
        descartados = int(n*pa)
        for i in range(descartados):
            i2 = ranking[i]
            # El nuevo nido se obtiene modificando uno de los nidos que no se pierden
            j = random.choice(ranking[descartados:])
            nuevo_camino = nidos[j]
            for _ in range(random.randint(1,5)): nuevo_camino = permutarConsecutivos(nuevo_camino)            
            nidos[i2] = nuevo_camino
            valores[i2] = g.evalCamino(nuevo_camino)
       
        # Buscamos el mejor valor
        best = min( range(n), key = lambda i: valores[i])
        if valores[best] < mejor_val:
            mejor_val = valores[best]
            mejor_sol = nidos[best]
    return mejor_sol, mejor_val


def CompleteCuckooSearch(g, n, it, pa, s0=1, alpha = 1, beta = 1.5, init = True):
    # Inicialización
    # Si init es True, entonces se inicializa desde Nearest Neighbour. En caso contrario, se inicializa desde una solución aleatoria
    nidos, valores = [], []
    mejor_val, mejor_sol = float('inf'), []
    for i in range(n):
        if init:
            # Inicialización desde Nearest Neighbour
            val, cam = NN_heuristic(g)
            nidos.append(cam)
            valores.append(val)
        else:
            # Inicialización aleatoria
            nodos = list(g.nodos.keys())
            x1 = random.sample(nodos, len(nodos))
            x1.append(x1[0])
            nidos.append(x1)
            valores.append(g.evalCamino(x1))
        
    for _ in range(it):
        # Todos los cucos llevan a cabo un levy flight
        for cuckoo in range(n):
            nuevo_camino = CuckooMove(nidos[cuckoo], alpha, beta, s0)
            val = g.evalCamino(nuevo_camino)
            if val < valores[cuckoo]:
                nidos[cuckoo] = nuevo_camino
                valores[cuckoo] = val
                
        # Una fraccion de los cucos es abandonada y se generan nuevos a partir de los que se conservan
        for cuckoo in range(n):
            r = random.random()
            if r < pa:
                # El nuevo nido se obtiene modificando uno de los nidos anteriores
                j = np.random.choice(n)
                nuevo_camino = nidos[j]
                for _ in range(random.randint(1,5)): nuevo_camino = permutarConsecutivos(nuevo_camino)
                val = g.evalCamino(nuevo_camino)
                if val < valores[i]:
                    nidos[i] = nuevo_camino
                    valores[i] = val
                
        # Buscamos el mejor valor
        best = min( range(n), key = lambda i: valores[i])
        if valores[best] < mejor_val:
            mejor_val = valores[best]
            mejor_sol = nidos[best]
    return mejor_sol, mejor_val



"""
Parte 3: Ant Colony System
    Heurística Nearest Neighbour (NN)
    Definición de las hormigas como clase
    Definición del algoritmo Ant Colony System (ACS)
    
"""

# Nearest Neighbour (NN)
# En cada paso se toma el nodo más cercano, partiendo de un nodo aleatorio
def NN_heuristic(grafo):
    camino, val = [], 0

    nodo_inicial = random.choice( list(grafo.nodos.keys()) )
    nodos = set(grafo.nodos.keys()) - {nodo_inicial}
    camino.append(nodo_inicial)
    
    nodo1 = nodo_inicial
    while len(nodos) > 0:
        minimo = float("inf")
        for nodo2 in nodos:
            aux = grafo.aristas[(nodo1, nodo2)]
            if aux < minimo:
                minimo = aux
                nodo_sig = nodo2
        camino.append(nodo_sig)
        nodo1 = nodo_sig
        val += minimo
        nodos.remove(nodo_sig)
    
    val += grafo.aristas[(camino[-1], nodo_inicial)]
    camino.append(nodo_inicial)
    return val, camino


# Definicion de las hormigas como clase
class Hormiga():
    def __init__(self, n, grafo, alpha = 1, beta = 2, q0 = 0.9, phi = 0.1):
        self.id = n
        self.grafo = grafo
        self.nodo_inicial = random.choice(list(grafo.nodos.keys()))
        self.nodos_visitados = [self.nodo_inicial]
        self.valor = 0
        self.actual = self.nodo_inicial
        self.disponibles = set(self.grafo.nodos.keys()) - {self.nodo_inicial}
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.phi = phi
        self.f0 = self.grafo.f0
        self.f = self.grafo.f
        self.aristas = self.grafo.aristas
    
    # Movimiento según la regla pseudo-aleatoria
    def movimiento(self):
        disp = list(self.disponibles)
        if len(disp) == 1:
            sig_id = disp[0]
        else:
            # Calculo de probabilidades
            feros = np.array([self.f[(self.actual, n2)] for n2 in disp])
            dists = np.array([self.aristas[(self.actual, n2)] for n2 in disp])
            ps = feros**self.alpha * (1/dists)**self.beta
            # Seleccion de nodo
            q = random.random()
            if q <= self.q0:
                sig = max(range(len(ps)), key = lambda x: ps[x])
            else:
                sig = np.random.choice(len(disp) , p= ps/np.sum(ps))
            sig_id = disp[sig]
            
        # Actualizacion de los datos de la hormiga
        arista_empleada = (self.actual, sig_id)
        self.nodos_visitados.append(sig_id)
        self.f[arista_empleada] = (1-self.phi)*self.f[arista_empleada]+ self.phi*self.f0
        if self.grafo.simetrico:
            self.f[ (sig_id, self.actual) ] = self.f[arista_empleada]
        self.valor += self.aristas[arista_empleada]
        self.actual = sig_id
        self.disponibles.remove(sig_id)
    
    # Generar una solución moviendose hasta completar el recorrido
    def generarSol(self):
        while len(self.disponibles) > 0:
            self.movimiento()
        ultima_arista = (self.actual, self.nodo_inicial)
        self.nodos_visitados.append(self.nodo_inicial)
        self.valor += self.aristas[ultima_arista]
        self.actual = self.nodo_inicial
    
    # Para obtener las aristas empeladas como un conjunto
    def recorrido(self):
        aristas_empleadas = set()
        for i, n1 in enumerate(self.nodos_visitados[:-1]):
            n2 = self.nodos_visitados[i+1]
            aristas_empleadas.add( (n1,n2) )
            if self.grafo.simetrico: aristas_empleadas.add( (n2,n1) )
        return aristas_empleadas


# Diccionario con las feromonas de cada arista
# Si no recibe valor de f0, lo calcula mediante 1/(n* L_NN)
def inicializarFeromonas(grafo, f0 = None):
    if f0 is None: f0 = 1/(len(grafo.nodos)*NN_heuristic(grafo)[0])
    grafo.f0 = f0
    grafo.f = {}
    for arista in grafo.aristas.keys():
        grafo.f[arista] = f0

# Ant Colony System (ACS)
def ACS(grafo, n = 100, it = 100, ro = 0.1, phi = 0.1, alpha = 1, beta = 2, q0 = 0.9, f0 = None):
    mejor_val = float('inf')
    inicializarFeromonas(grafo, f0)
    for i in range(it):
        # Generar las hormigas
        for j in range(n):
            h = Hormiga(j, grafo, alpha, beta, q0, phi)
            h.generarSol()
            if h.valor < mejor_val:
                mejor_h = h
                mejor_val = h.valor
        # Actualizar las feromonas de forma global
        mejor_recorrido = h.recorrido()
        for arista in grafo.aristas.keys():
            aux = 1/(mejor_val + 10**-8) if arista in mejor_recorrido else 0
            grafo.f[arista] = (1-ro)* grafo.f[arista] + ro*aux
        # La actualizacion local tiene lugar durante el movimieno de la hormiga
    return mejor_h.nodos_visitados, mejor_val




"""
 Parte 4: Ejecución de los algoritmos para diferentes problemas
"""

Problemas = ["eil51", "berlin52", "eil76",
              "kroA100", "kroB100", "kroC100",
              "kroD100", "kroE100", "eil101",
              "lin105","bier127","ch130",
              "ch150", "kroA150", "kroB150", 
              "kroA200", "kroB200", "lin318"]

Optimos = [ 426, 7542, 538,
           21282, 22141, 20749,
           21294, 22068, 629,
           14379, 118282, 6110,
           6528, 26524, 26130,
           29368, 29437, 42029]

Algoritmos = [CuckooSearchMejorado,
              CuckooSearchMejorado,
              CompleteCuckooSearch,
              ACS]

Parametros = [{"n": 5,   "it": 50000, "pa":0.25, "pc": 0,   "s0": 1,  "alpha":1, "beta": 1.5},
              {"n": 50,  "it": 4000,  "pa":0.25, "pc": 0.5, "s0": 1,  "alpha":1, "beta": 1.5},
              {"n": 25 , "it": 4000,  "pa":0.25, "s0": 1,   "alpha":1,"beta":1.5},
              {"n": 50,  "it": 200 ,  "ro": 0.1, "phi":0.1, "alpha":1,"beta": 2, "q0":0.9}]

Titulos = ["Cuckoo Search: ",
           "Improved Cuckoo Search: ",
           "Complete Cuckoo Search: ",
           "Ant Colony System: "]


# Ejecución de un algoritmo dados unos parámetros y durante un número de repeticiones
# Si graficas es True, entonces representa el mejor camino encontrado
def obtenerResultados(g, algoritmo, parametros, titulo, repeticiones = 30, graficas = False):
    valores, tiempos = [], []
    mejor_val = float('inf')
    for i in range(repeticiones):
        t = time.time()
        camino, val = algoritmo(g, **parametros)
        valores.append(val)
        tiempos.append(time.time()-t)
        # print(val, f" ,  tiempo = {time.time()-a1}")
        if val < mejor_val:
            veces_optimo = 1
            mejor_camino = camino
            mejor_val = val
        elif val == mejor_val:
            veces_optimo += 1
    media = np.mean(valores)
    std = np.std(valores)
    
    if graficas:
        g.plot(mejor_camino)
        plt.title(titulo + f"Valor: {mejor_val:.1f}, Valor Medio: {media:.1f}")
        plt.show()
        
    return {"t" : np.mean(tiempos),
            "media" : media,
            "std" : std,
            "mejor" : mejor_val,
            "camino" : mejor_camino,
            "optimos" : veces_optimo}


# Funciones para probar cada uno de los algoritmos individualmente
# Empleadas para probar cada uno de los hiperparámetros
def mainCSM(name, parametros = Parametros[1], repes = 10, prints =True, graficas = False):
    file = "tsp/" + name + ".tsp"
    g = leerTSP(file)
    print(*parametros.values())
    a = time.time()
    if parametros["pc"] == 0: titulo = 'Cuckoo Search: '
    else: titulo = 'Improved Cuckoo Search: '
    resultado = obtenerResultados(g, CuckooSearchMejorado, parametros, titulo, repes, graficas) 
    if prints: print(titulo,
                     f"Mejor: {resultado['mejor']:.1f}",
                     f"Valor Medio: {resultado['media']:.1f}",
                     f"Desviación Típica: {resultado['std']:.1f}",
                     f"Tiempo por ejecución: {resultado['t']:.1f}",
                     sep = "\n", end = "\n")
    print("Tiempo Total: ", time.time()-a)
    return resultado


def mainCCS(name, parametros = Parametros[2], repes = 10, prints =True, graficas = False):
    file = "tsp/" + name + ".tsp"
    g = leerTSP(file)
    print(*parametros.values())
    a = time.time()
    titulo = 'Complete Cuckoo Search: '
    resultado = obtenerResultados(g, CompleteCuckooSearch, parametros, titulo, repes, graficas) 
    if prints: print(titulo,
                     f"Mejor: {resultado['mejor']:.1f}",
                     f"Valor Medio: {resultado['media']:.1f}",
                     f"Desviación Típica: {resultado['std']:.1f}",
                     f"Tiempo por ejecución: {resultado['t']:.1f}",
                     sep = "\n", end = "\n")
    print("Tiempo Total: ", time.time()-a)
    return resultado


def mainACS(name, parametros = Parametros[3], repes = 10, prints =True, graficas = False):
    file = "tsp/" + name + ".tsp"
    g = leerTSP(file)
    print(*parametros.values())
    a = time.time()
    titulo = 'Ant Colony System: '
    resultado = obtenerResultados(g, ACS, parametros, titulo, repes, graficas) 
    if prints: print(titulo,
                     f"Mejor: {resultado['mejor']:.1f}",
                     f"Valor Medio: {resultado['media']:.1f}",
                     f"Desviación Típica: {resultado['std']:.1f}",
                     f"Tiempo por ejecución: {resultado['t']:.1f}",
                     sep = "\n", end = "\n")
    print("Tiempo Total: ", time.time()-a)
    return resultado



# Devuelve un diccionario con los resultados por algoritmo y por problema
# Los parametros graficas y prints permiten controlar si queremos imprimir los resultados individuales de cada algoritmo
def obtenerDatos(repeticiones = 30, prints = True, graficas = True):
    datos = { algoritmo[:-2] : {} for algoritmo in Titulos }
    for indice, name in enumerate(Problemas):
        if prints: print(f"\n\n\n{name}\nValor Óptimo: {Optimos[indice]}\n")
        file = "tsp/" + name + ".tsp"
        g = leerTSP(file)
        for i in range(len(Algoritmos)):
            aux = obtenerResultados(g, Algoritmos[i], Parametros[i], Titulos[i], repeticiones, graficas)
            datos[Titulos[i][:-2]][name] = aux
            if prints: print(Titulos[i],
                             f"Mejor: {aux['mejor']:.1f}",
                             f"Valor Medio: {aux['media']:.1f}",
                             f"Desviación Típica: {aux['std']:.1f}",
                             f"Tiempo por ejecución: {aux['t']:.1f}",
                             sep = "\n", end = "\n\n")
    return datos


# Obtener la tabla con los resultados en formato LaTeX
def latex(repeticiones = 30, datos = None):
    tabla = "\\begin{tabular}{!{\\vrule width 1.5pt}c|c!{\\vrule width 1.5pt}c|c|c|c!{\\vrule width 1.1pt}c|c|c|c!{\\vrule width 1.1pt}c|c|c|c!{\\vrule width 1.1pt}c|c|c|c!{\\vrule width 1.5pt}}\n"
    tabla += "\\Xhline{1.2pt}\n"
    tabla += "\\multirow{2}{*}{Instancia} & \\multirow{2}{*}{Óptimo} & \\multicolumn{4}{c!{\\vrule width 1.1pt}}{CS} & \\multicolumn{4}{c!{\\vrule width 1.1pt}}{ICS} & \\multicolumn{4}{c!{\\vrule width 1.1pt}}{CCS} & \\multicolumn{4}{c!{\\vrule width 1.5pt}}{ACS} \\\\\n"
    tabla += "\\cline{3-18}\n"
    tabla += "& & Mejor & Media & STD & Opts & Mejor & Media & STD & Opts & Mejor & Media & STD & Opts & Mejor & Media & STD & Opts \\\\\n"
    tabla += "\\Xhline{1.2pt}\n"
    
    if datos is None: 
        sacarDatos = True
        datos = { algoritmo[:-2] : {} for algoritmo in Titulos }
    else: sacarDatos = False
    
    for indice, name in enumerate(Problemas):           
        tabla += f"{name} & {Optimos[indice]}"
        print(name)
        
        if sacarDatos:    
            file = "tsp/" + name + ".tsp"
            g = leerTSP(file)
        
        for i in range(len(Algoritmos)):
            alg = Titulos[i][:-2]
            print(alg)
            if sacarDatos: 
                resultados = obtenerResultados(g, Algoritmos[i], Parametros[i], Titulos[i], repeticiones)
                datos[alg][name] = resultados
            else: resultados = datos[alg][name]
            
            if int(resultados["mejor"]) == Optimos[indice]:
                tabla += f"& \\textbf{{ {resultados['mejor']:.0f} }} "
            else: tabla += f"& {resultados['mejor']:.0f} "
            
            if int(resultados["media"]) == Optimos[indice]:
                tabla += f"& \\textbf{{ {resultados['media']:.0f} }} "
            else: tabla += f"& {resultados['media']:.1f} "
            
            if int(resultados["std"]) == 0:
                tabla += f"& \\textbf{{ {resultados['std']:.1f} }} "
            else: tabla += f"& {resultados['std']:.1f} "
            
            if resultados["mejor"] == Optimos[indice]:
                tabla += f"& {resultados['optimos']:.1f} "
            else: tabla += "& 0 "
            
        if indice != len(Problemas) -1:
            tabla += "\\\\\n\\hline\n"
        else: tabla += "\\\\\n\\Xhline{1.2pt}\n"
        print("\n")
    tabla += "\\end{tabular}\n"
    print(tabla)
    return tabla, datos
      

# Representa los resultados en una grafica con respecto el PDav%
def comparacionGrafica(datos, problemas = Problemas, optimos = Optimos[:len(Problemas)]):
    markers = ['o', 's', 'D', '^']
    colors = ['red', 'blue', 'orange', 'purple']
    linestyles = ['-', '--', ':', '-.']
    plt.figure(figsize=(15, 7))
    for i, alg in enumerate(datos.keys()):
        resultados = [ datos[alg][p]['media'] for p in problemas]
        PDav = [ (resultados[i]- opt)/opt*100 for i, opt in enumerate(optimos)]
        
        plt.scatter(Problemas, PDav, label = alg, marker = markers[i], color = colors[i], s=80)
        plt.plot(Problemas, PDav, color = colors[i], linestyle=linestyles[i], linewidth=2)
    plt.xlabel('Problemas')
    plt.ylabel('PDav (%)')
    plt.title('Comparación de los resultados según PDav (%)')
    plt.legend()
    
    plt.show()


# Obtener los resultados para la tabla y la grafica
def main(repeticiones = 30):
    datos = obtenerDatos(repeticiones, False, False)
    tabla, datos = latex(repeticiones, datos)
    comparacionGrafica(datos)
    # plt.savefig("resultados.png")