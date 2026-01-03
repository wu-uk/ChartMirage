import os
import json
import qdrant_client
from llama_index.core import StorageContext, Document, load_index_from_storage
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import ImageDocument

from core.experiments.pipeline import DualModalPipeline

class QdrantPipeline(DualModalPipeline):
    """
    Pipeline using Qdrant for vector storage (Text + Image).
    Inherits from DualModalPipeline to reuse logic, but overrides index building/loading.
    """
    def __init__(self, dataset_path, storage_dir, eval_enabled=True, image_key="image", caption_key="original_caption"):
        super().__init__(dataset_path, storage_dir, eval_enabled, image_key, caption_key)
        # Qdrant DB path inside the storage directory
        self.qdrant_path = os.path.join(self.storage_dir, "qdrant_db")

    def build_index(self, force_rebuild=False):
        # Ensure storage directory exists
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        # Initialize Qdrant Client
        client = qdrant_client.QdrantClient(path=self.qdrant_path)

        # Initialize Vector Stores
        text_store = QdrantVectorStore(
            client=client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=client, collection_name="image_collection"
        )

        # Check if we can load existing index
        # We check for docstore.json in storage_dir as a proxy for existence
        index_exists = os.path.exists(os.path.join(self.storage_dir, "docstore.json"))

        if not force_rebuild and index_exists:
            print(f"Loading existing Qdrant index from {self.storage_dir}...")
            # Re-construct storage context with existing persistence + Qdrant stores
            storage_context = StorageContext.from_defaults(
                persist_dir=self.storage_dir,
                vector_store=text_store,
                image_store=image_store
            )
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
        else:
            print(f"Building new Qdrant index (DualModal: {self.image_key} + {self.caption_key})...")
            
            # Create fresh StorageContext
            storage_context = StorageContext.from_defaults(
                vector_store=text_store,
                image_store=image_store
            )

            # Load Data (Copied/Adapted from DualModalPipeline)
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            
            for entry in data:
                img_rel_path = entry.get(self.image_key)
                caption = entry.get(self.caption_key)
                
                if not img_rel_path:
                    continue
                
                img_abs_path = self._resolve_image_path(img_rel_path)
                
                if not img_abs_path:
                    # print(f"Warning: Image not found for {img_rel_path}")
                    continue

                # 1. ImageDocument
                img_doc = ImageDocument(
                    image_path=img_abs_path,
                    text=caption if caption else "", 
                    metadata={
                        "file_name": os.path.basename(img_abs_path),
                        "type": "image_node",
                        "doc_id": img_rel_path,
                        "caption_type": self.caption_key,
                        "image_type": self.image_key
                    }
                )
                documents.append(img_doc)
                
                # 2. TextDocument
                if caption:
                    text_doc = Document(
                        text=caption,
                        metadata={
                            "related_image": os.path.basename(img_abs_path),
                            "type": "text_node",
                            "doc_id": img_rel_path,
                            "caption_type": self.caption_key,
                            "image_type": self.image_key
                        }
                    )
                    documents.append(text_doc)
            
            print(f"Created {len(documents)} documents.")
            print("Indexing into Qdrant...")
            
            self.index = MultiModalVectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
            
            # Persist the docstore/index_store to disk (vectors are in Qdrant)
            print(f"Persisting storage context to {self.storage_dir}...")
            self.index.storage_context.persist(persist_dir=self.storage_dir)

        # Initialize Query Engine
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=1,
            image_similarity_top_k=1
        )
        print("QdrantPipeline initialized.")
