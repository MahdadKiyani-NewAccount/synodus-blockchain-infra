import faiss
import numpy as np

# Genereer een dummy dataset van 100 vectors van 128 dimensies
dim = 128
nb = 100
data = np.random.random((nb, dim)).astype('float32')

# FAISS index maken en vullen
index = faiss.IndexFlatL2(dim)
index.add(data)

# Opslaan
faiss.write_index(index, "faiss_index.index")
print("FAISS index opgeslagen als faiss_index.index")

