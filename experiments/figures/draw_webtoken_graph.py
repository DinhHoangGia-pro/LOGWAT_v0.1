import networkx as nx
import matplotlib.pyplot as plt

def draw_web_attack_topology():
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    G_attack = nx.DiGraph()
    atk_nodes = ['SELECT', '*', 'FROM', 'users', 'WHERE', 'id', 'OR', '1=1', '--']
    G_attack.add_nodes_from(atk_nodes)
    
    seq_edges = [('SELECT', '*'), ('*', 'FROM'), ('FROM', 'users'), ('users', 'WHERE'), 
                 ('WHERE', 'id'), ('id', 'OR'), ('OR', '1=1'), ('1=1', '--')]
    G_attack.add_edges_from(seq_edges)
    
    skip_edges = [('SELECT', 'FROM'), ('FROM', 'WHERE'), ('WHERE', 'OR'), ('OR', '--')]
    G_attack.add_edges_from(skip_edges)
    
    pos1 = nx.spring_layout(G_attack, seed=42)
    
    nx.draw_networkx_edges(G_attack, pos1, edgelist=seq_edges, ax=ax1, 
                           edge_color='gray', width=1.5, arrowsize=20)
    
    nx.draw_networkx_edges(G_attack, pos1, edgelist=skip_edges, ax=ax1, 
                           edge_color='red', width=2, style='dashed', arrowsize=25)
    
    nx.draw_networkx_nodes(G_attack, pos1, ax=ax1, node_color='white', 
                           edgecolors='red', linewidths=2, node_size=3500)
    nx.draw_networkx_labels(G_attack, pos1, ax=ax1, font_size=9, font_weight='bold')
    
    ax1.set_title("(a) SQL Injection Attack Graph\n(Red Dashed: Skip Edges)", fontsize=14, pad=20)

    G_normal = nx.DiGraph()
    norm_nodes = ['GET', '/index.html', 'HTTP/1.1', 'Host:', 'google.com', 'Accept:']
    G_normal.add_nodes_from(norm_nodes)
    norm_edges = [('GET', '/index.html'), ('/index.html', 'HTTP/1.1'), 
                  ('HTTP/1.1', 'Host:'), ('Host:', 'google.com'), ('google.com', 'Accept:')]
    G_normal.add_edges_from(norm_edges)

    pos2 = nx.shell_layout(G_normal) 
    
    nx.draw(G_normal, pos2, ax=ax2, with_labels=True, 
            node_color='white', edgecolors='green', linewidths=2, 
            node_size=3500, font_size=9, font_weight='bold', 
            edge_color='gray', arrowsize=20, width=1.5)
    
    ax2.set_title("(b) Normal HTTP Request\nLinear Sequence Topology", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig("Web_Attack_Comparison.svg", format='svg', bbox_inches='tight', dpi=300)
    print("Đã xuất file Web_Attack_Comparison.svg thành công!")
    plt.show()

if __name__ == "__main__":
    draw_web_attack_topology()
