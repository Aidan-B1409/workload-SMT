import json
import random
import os

def generate_synthetic_workflows():
    # Define worker capacities to calculate compatible_workers
    workers_capacity = {
        "laptop": {"CPU": 2, "GPU": 0},
        "workstation": {"CPU": 8, "GPU": 0},
        "cloud_cpu": {"CPU": 8, "GPU": 0},
        "cloud_gpu_1": {"CPU": 8, "GPU": 1},
        "cloud_gpu_2": {"CPU": 8, "GPU": 1}
    }
    
    available_workers = list(workers_capacity.keys())
    
    # We will generate 5 synthetic tasks with larger sizes (27 to 69 nodes) and larger proc_time requirements
    task_configs = [
        {"id": "syn_ml-pipeline-027", "parts": 2, "folds": 3},
        {"id": "syn_ml-pipeline-035", "parts": 2, "folds": 5},
        {"id": "syn_ml-pipeline-046", "parts": 3, "folds": 4},
        {"id": "syn_ml-pipeline-052", "parts": 3, "folds": 5},
        {"id": "syn_ml-pipeline-069", "parts": 4, "folds": 5},
    ]
    
    workflows = []
    
    for config in task_configs:
        task_id = config["id"]
        parts = config["parts"]
        folds = config["folds"]
        
        nodes = []
        edges = []
        
        def add_node(nid, name, desc, pt, req):
            # compute compatible workers based on requirements
            compat = []
            for wid, cap in workers_capacity.items():
                if cap["CPU"] >= req.get("CPU", 0) and cap["GPU"] >= req.get("GPU", 0):
                    compat.append(wid)
            nodes.append({
                "id": nid,
                "name": name,
                "description": desc,
                "processing_time": pt,
                "requires": req,
                "compatible_workers": compat
            })
            
        def add_edge(src, dst, data):
            edges.append({
                "from": src,
                "to": dst,
                "data": [data]
            })
            
        # 1. Ensemble node (sink)
        ensemble_id = "ensemble_predictions"
        # We add it at the end, but let's define its parameters
        ensemble_pt = random.randint(150, 300)
        ensemble_req = {"CPU": 4, "GPU": 0}
        
        # Build DAG
        for p in range(parts):
            load_id = f"load_data_part_{p}"
            clean_id = f"clean_data_part_{p}"
            impute_id = f"impute_data_part_{p}"
            feat_num_id = f"extract_numeric_part_{p}"
            feat_txt_id = f"extract_text_part_{p}"
            feat_img_id = f"extract_image_part_{p}"
            fuse_id = f"fuse_features_part_{p}"
            
            add_node(load_id, f"Load Data Part {p}", f"Load subset of input raw data for part {p}", random.randint(50, 100), {"CPU": 1, "GPU": 0})
            add_node(clean_id, f"Clean Data Part {p}", f"Standardize columns and clean values for part {p}", random.randint(60, 120), {"CPU": 1, "GPU": 0})
            add_node(impute_id, f"Impute Missing Part {p}", f"Impute missing features for part {p}", random.randint(60, 120), {"CPU": 2, "GPU": 0})
            add_node(feat_num_id, f"Extract Numeric Features Part {p}", f"Scale and extract statistical features for part {p}", random.randint(80, 180), {"CPU": 2, "GPU": 0})
            add_node(feat_txt_id, f"Extract Text Features Part {p}", f"Apply TF-IDF / SVD to text fields for part {p}", random.randint(150, 350), {"CPU": 4, "GPU": 0})
            add_node(feat_img_id, f"Extract Image Features Part {p}", f"Run ResNet/ViT feature extractor for part {p}", random.randint(250, 500), {"CPU": 4, "GPU": 1})
            add_node(fuse_id, f"Fuse Features Part {p}", f"Combine numeric, text, and image features for part {p}", random.randint(80, 160), {"CPU": 2, "GPU": 0})
            
            add_edge(load_id, clean_id, f"raw_data_part_{p}")
            add_edge(clean_id, impute_id, f"cleaned_data_part_{p}")
            add_edge(impute_id, feat_num_id, f"imputed_data_part_{p}")
            add_edge(impute_id, feat_txt_id, f"imputed_data_part_{p}")
            add_edge(impute_id, feat_img_id, f"imputed_data_part_{p}")
            add_edge(feat_num_id, fuse_id, f"numeric_features_part_{p}")
            add_edge(feat_txt_id, fuse_id, f"text_features_part_{p}")
            add_edge(feat_img_id, fuse_id, f"image_features_part_{p}")
            
            for f in range(folds):
                train_id = f"train_model_part_{p}_fold_{f}"
                eval_id = f"evaluate_model_part_{p}_fold_{f}"
                
                # CPU or GPU model training
                train_req = random.choice([{"CPU": 8, "GPU": 0}, {"CPU": 8, "GPU": 1}])
                add_node(train_id, f"Train Model Part {p} Fold {f}", f"Train GBDT or deep neural net on fold {f} of part {p}", random.randint(300, 800), train_req)
                add_node(eval_id, f"Evaluate Model Part {p} Fold {f}", f"Validate model predictions on validation fold {f} of part {p}", random.randint(100, 250), {"CPU": 2, "GPU": 0})
                
                add_edge(fuse_id, train_id, f"fused_features_part_{p}")
                add_edge(train_id, eval_id, f"model_part_{p}_fold_{f}")
                add_edge(eval_id, ensemble_id, f"predictions_part_{p}_fold_{f}")
                
        # Finally add the ensemble node
        add_node(ensemble_id, "Ensemble Predictions", "Perform final soft-voting ensemble across all folds and parts", ensemble_pt, ensemble_req)
        
        workflows.append({
            "task_id": task_id,
            "task_description": f"A synthetic scalable machine learning pipeline with {parts} parts and {folds} folds to test larger schedules.",
            "dag_nodes": nodes,
            "dag_edges": edges,
            "workers": available_workers
        })
        
    os.makedirs("data", exist_ok=True)
    with open("data/synthetic_workflows.json", "w") as f:
        json.dump(workflows, f, indent=2)
        
    print(f"Successfully generated {len(workflows)} synthetic workflows and saved to data/synthetic_workflows.json")

if __name__ == "__main__":
    generate_synthetic_workflows()
