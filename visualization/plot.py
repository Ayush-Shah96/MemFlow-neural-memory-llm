from __future__ import annotations

import math
import networkx as nx
import plotly.graph_objects as go
from graph.store import KnowledgeGraph


def build_graph_figure(graph: KnowledgeGraph, highlight_nodes: set[str] | None = None):
    g = graph.graph
    if g.number_of_nodes() == 0:
        return go.Figure().update_layout(title="No graph data yet")
    highlight_nodes = highlight_nodes or set()
    layout = nx.spring_layout(g.to_undirected(), seed=42, k=max(0.3, 1 / math.sqrt(max(1, g.number_of_nodes()))))
    edge_x=[]; edge_y=[]; edge_text=[]
    for u,v,data in g.edges(data=True):
        x0,y0=layout[u]; x1,y1=layout[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
        edge_text.append(data.get("relation","related_to"))
    edge_trace = go.Scatter(x=edge_x,y=edge_y,mode="lines",line={"width":1},hoverinfo="skip")
    node_x=[]; node_y=[]; labels=[]; hovers=[]
    for n,data in g.nodes(data=True):
        x,y=layout[n]; node_x.append(x); node_y.append(y); labels.append(data.get("name",n))
        hovers.append(f"{data.get('name',n)}<br>Type: {data.get('type','Concept')}<br>Chunks: {len(data.get('chunks',[]))}")
    node_trace = go.Scatter(x=node_x,y=node_y,mode="markers+text",text=labels,textposition="top center",hovertext=hovers,hoverinfo="text",marker={"size":[20 if n in highlight_nodes else 12 for n in g.nodes()]})
    # Edge labels are shown as a compact annotation set.
    annotations=[]
    for (u,v,data) in g.edges(data=True):
        x=(layout[u][0]+layout[v][0])/2; y=(layout[u][1]+layout[v][1])/2
        annotations.append(dict(x=x,y=y,text=data.get("relation","related_to"),showarrow=False,font={"size":9}))
    return go.Figure([edge_trace,node_trace]).update_layout(title="Persistent Knowledge Graph",showlegend=False,hovermode="closest",annotations=annotations,margin=dict(l=10,r=10,t=40,b=10),xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
