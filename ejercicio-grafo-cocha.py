import osmnx as ox
import networkx as nx

import matplotlib.pyplot as pl
import geopy

from geopy.geocoders import Nominatim

#llamamos la erramienta de nomatim y creamos su clase
#loc = Nominatim(user_agent="Geopy Library")
#ingresamos el nombre la ciudad
#get

if __name__ == "__main__":
    ciudad = "Cochabamba, Bolivia"
    grafo = ox.graph_from_place(ciudad, network_type="drive")

    fig, ax = ox.plot_graph(grafo, node_size=10, node_color = "red", edge_color='black', edge_linewidth=0.5)

    ##google tedevueve en latitud longitud
    #cato = -17.393542, -66.146135
    #chanchimon = -17.371665344019387, -66.14374891947371
    punto_inicial = (-17.393542, -66.146135)
    punto_final = (-17.371665344019387, -66.14374891947371)

    nodo_inicial = ox.distance.nearest_nodes(grafo, punto_inicial[1], punto_inicial[0])
    nodo_final = ox.distance.nearest_nodes(grafo, punto_final[1], punto_final[0])


    ruta = nx.shortest_path(grafo, nodo_inicial, nodo_final, weight='length')
    fig, ax = ox.plot_graph_route(grafo, ruta, route_linewidth=4, node_size=30, node_color="blue", route_color="yellow")
    print("numero de nodos encontrados: ", grafo.number_of_nodes())
    print("numero de conexiones: ", grafo.number_of_edges())
    ox.save_graphml(grafo, "calles_cbba.graphml")