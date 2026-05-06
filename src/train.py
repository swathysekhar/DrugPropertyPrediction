
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data_loader import (load_data_long, CustomDataset, adj_list_to_adj_matrix )
from src.graph_model import GraphTransformer
from src.smiles_model import SMILESTransformer
from src.functional_groups import FUNCTIONAL_GROUPS, get_functional_group_dim
from src.fg_model import FunctionalGroupEmbedding
from src.fusion_model import FusionModel


def main(data_name,options):

    vocab_size = 100
    d_model = 128
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512
    max_length = 100
    batch_size = 1
    num_epochs = 100
    embed_dim = 128

    train_data, train_labels, test_data, test_labels = load_data_long(data_name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, train_labels, test_data, test_labels = load_data_long(data_name, device)

    # convert train_labels to numpy
    train_labels_np = train_labels.cpu().numpy() if torch.is_tensor(train_labels) else train_labels
    test_labels_np = test_labels  # already numpy in your code

    print("Train:", np.bincount(train_labels_np.astype(int)))
    print("Test :", np.bincount(test_labels_np.astype(int)))
    input_dim_train = train_data['features'][0].size(-1)
    input_dim_test = test_data['features'][0].size(-1)


    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]


    data_list_train = [Data(x=features.clone().detach(),
                              edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                              y=torch.tensor(label, dtype=torch.float))
                         for features, adj_matrix, label in zip(train_data['features'], adj_matrices_train, train_labels)]
    data_list_test = [Data(x=features.clone().detach(),
                                edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                                y=torch.tensor(label, dtype=torch.float))
                            for features, adj_matrix, label in zip(test_data['features'], adj_matrices_test, test_labels)]

    train_dataset = CustomDataset(data_list_train, train_data['sequence'], train_data['fg'])
    test_dataset = CustomDataset(data_list_test, test_data['sequence'], test_data['fg'])

    if options[0]:
        graph_model = GraphTransformer(in_channels=input_dim_train, hidden_channels=64, embed_dim=embed_dim, heads=4).to(device)
    if options[1]:
        sequence_model = SMILESTransformer(vocab_size, embed_dim, nhead, num_encoder_layers, dim_feedforward,max_length=100).to(device)
    if options[2]:
        fg_encoder = FunctionalGroupEmbedding(fg_input_dim=len(FUNCTIONAL_GROUPS), embed_dim=embed_dim).to(device)
    if options[3]:
        fusion_model = FusionModel(graph_model, sequence_model, fg_encoder, embed_dim=embed_dim).to(device)
        #fusion_model = FusionModel(graph_model, sequence_model).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001, weight_decay=5e-4)


    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    session_name = f'{data_name}_{formatted_datetime}'
    folder_path = os.path.join('saved_models', session_name)
    os.makedirs(folder_path, exist_ok=True)

    output_dir_train = f'output/{data_name}/train'
    os.makedirs(output_dir_train, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name_train = f'{output_dir_train}/train_accuracy_details_{current_time}.txt'

    output_dir_test = f'output/{data_name}/test'
    os.makedirs(output_dir_test, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name_test = f'{output_dir_test}/test_accuracy_details_{current_time}.txt'


    best_train_accuracy = 0.0
    best_test_accuracy=0.0
    # Training loop

    with open(file_name_train, 'a') as file_train, open(file_name_test, 'a') as file_test:
        for epoch in range(100):
            total_correct = 0
            total_samples = 0
            true_labels_train = []
            pred_probs_train = []
            losses=0.0

            for data_batch in train_dataset:
                graph_data_batch = data_batch[0]
                sequence_inputs  = data_batch[1]
                fg = data_batch[2]
                if len(fg.shape) == 1:
                  fg = fg.unsqueeze(0)
                fg = fg.to(device)
                sequence_targets=graph_data_batch.y

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output, attn_weights = fusion_model(graph_data_batch, sequence_inputs,fg)

                # Compute binary predictions
                prob = torch.sigmoid(output)
                binary_predictions = (prob >= 0.5).float()
                
                # Move and reshape sequence_targets BEFORE comparison
                sequence_targets = sequence_targets.view(-1, 1).to(device)

                # Compute batch accuracy
                batch_correct = (binary_predictions == sequence_targets).sum().item()
                total_samples += 1 #total_samples should be 1 since batch_size is 1
                total_correct += batch_correct # Add batch_correct after reshape


                true_labels_train.append(sequence_targets.cpu().numpy().reshape(-1))
                pred_probs_train.append(prob.detach().cpu().numpy()[0]) # Ensure 1D array...............
                #print(output,sequence_targets,pred_probs_train)

                # Compute loss

                loss = criterion(output, sequence_targets)
                losses+=loss.item()

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()


            # Compute epoch accuracy
            epoch_train_accuracy = (total_correct / total_samples)*100
            print(f"Epoch {epoch+1}/{100}, Epoch Accuracy: {epoch_train_accuracy:.4f}")

            if epoch_train_accuracy >= best_train_accuracy:
                best_train_accuracy = epoch_train_accuracy
                model_path = os.path.join(folder_path, f'train_best_model_{best_train_accuracy:.3f}.pth')
                torch.save(fusion_model.state_dict(), model_path)
                print("Saved model with accuracy train model with accuracy{:.2f}% to {}".format(best_train_accuracy, model_path))

            true_labels_train = np.concatenate(true_labels_train).astype(int) # Convert to int
            pred_probs_train = np.nan_to_num(np.concatenate(pred_probs_train), nan=0)
            predicted_labels_train_int = (pred_probs_train >= 0.5).astype(int)

            precision_train = precision_score(true_labels_train, predicted_labels_train_int)
            recall_train = recall_score(true_labels_train, predicted_labels_train_int)
            auc_roc_train = roc_auc_score(true_labels_train, pred_probs_train)
            f1_train = f1_score(true_labels_train, predicted_labels_train_int)
            print(f"Train AUC-ROC: {auc_roc_train:.4f}, Train F1 Score: {f1_train:.4f} , Train Precision: {precision_train:.4f}, Train Recall: {recall_train:.4f}\n")
            file_train.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Train AUC-ROC: {auc_roc_train:.4f}, Train F1 Score: {f1_train:.4f} , Train Precision: {precision_train:.4f}, Train Recall: {recall_train:.4f}\n')

            total_correct = 0
            total_samples = 0
            true_labels_test = []
            pred_probs_test = []

            for data_batch in test_dataset:
                graph_data_batch = data_batch[0]
                sequence_inputs = data_batch[1]
                fg = data_batch[2]
                if len(fg.shape) == 1:
                  fg = fg.unsqueeze(0)
                fg = fg.to(device)
                sequence_targets = graph_data_batch.y


                output, attn_weights = fusion_model(graph_data_batch, sequence_inputs,fg)
                prob = torch.sigmoid(output)
                binary_predictions = (prob >= 0.5).float()#......................
                #binary_predictions = (output >= 0.5).float()

                # Move and reshape sequence_targets BEFORE comparison
                sequence_targets = sequence_targets.view(-1, 1).to(device)

                batch_correct = (binary_predictions == sequence_targets).sum().item()
                total_samples += 1 #total_samples should be 1 since batch_size is 1
                total_correct += batch_correct

                true_labels_test.append(sequence_targets.cpu().numpy().reshape(-1))
                pred_probs_test.append(prob.detach().cpu().numpy()[0])#...............

            epoch_test_accuracy = (total_correct / total_samples)*100
            print(f"Epoch Testing Accuracy : {epoch_test_accuracy:.4f}")

            if epoch_test_accuracy >= best_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                model_path = os.path.join(folder_path, f'test_best_model_{best_test_accuracy:.3f}.pth')
                torch.save(fusion_model.state_dict(), model_path)
                print("Saved model with Test Model with accuracy {:.2f}% to {}".format(best_test_accuracy, model_path))


            true_labels_test = np.concatenate(true_labels_test).astype(int) # Convert to int
            pred_probs_test = np.nan_to_num(np.concatenate(pred_probs_test), nan=0)
            predicted_labels_test_int = (pred_probs_test >= 0.5).astype(int)

            print("Prob min:", pred_probs_test.min())
            print("Prob max:", pred_probs_test.max())
            print("Prob mean:", pred_probs_test.mean())
            print("Pred counts:", np.bincount(predicted_labels_test_int))

            precision_test = precision_score(true_labels_test, predicted_labels_test_int)
            recall_test = recall_score(true_labels_test, predicted_labels_test_int)
            auc_roc_test = roc_auc_score(true_labels_test, pred_probs_test)
            f1_test = f1_score(true_labels_test, predicted_labels_test_int)
            print(f"Test AUC-ROC: {auc_roc_test:.4f}, Test F1 Score: {f1_test:.4f}, Test Precision: {precision_test:.4f}, Test Recall: {recall_test:.4f}\n")
            file_test.write(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {epoch_test_accuracy:.4f},Test AUC-ROC: {auc_roc_test:.4f}, Test F1 Score: {f1_test:.4f}, Test Precision: {precision_test:.4f}, Test Recall: {recall_test:.4f} \n')
    file_test.close()
    file_train.close()




if __name__ == "__main__":
    options=[True,True,True,True]
    data_name='bace'
    main(data_name,options)