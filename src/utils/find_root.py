import os

def get_project_root() -> str:
    """
    Encontra o root do projeto procurando pelo setup.py.
    
    Returns:
        Caminho absoluto para o root do projeto
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'setup.py')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Não foi possível encontrar o root do projeto (setup.py não encontrado)")