#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

class TDFRONTDatasetLoader:
    def __init__(self):
        self.reset()

    def reset(self):
        self.dataset_path = None
        self.model_rootpath = None
        self.output_dataset_path = None
        self.target_model_label_and_path_list = []
        return

    def loadDatasetPath(self, dataset_path, model_rootpath):
        self.reset()
        self.dataset_path = dataset_path
        self.model_rootpath = model_rootpath
        return

    def checkKeyExists(self, json_object, key):
        if json_object is None:
            return False
        if key in json_object.keys():
            return True
        return False

    def getValue(self, json_object, key):
        if not self.checkKeyExists(json_object, key):
            return None
        return json_object[key]

    def getFurnitureList(self):
        source_json_file_name_list = os.listdir(self.dataset_path)

        target_class_key = "furniture"
        target_label_key = "title"

        furniture_list = []

        detected_source_json_num = 0

        for source_json_file_name in source_json_file_name_list:
            detected_source_json_num += 1
            source_json_file_path = self.dataset_path + source_json_file_name

            with open(source_json_file_path, "r") as f:
                current_source_json = json.load(f)

                if not self.checkKeyExists(current_source_json, target_class_key):
                    continue

                target_model_json_array = current_source_json[target_class_key]

                for target_model_json in target_model_json_array:
                    if not self.checkKeyExists(target_model_json, 'jid'):
                        continue
                    target_model_folder_name = target_model_json['jid']
                    target_model_folder_path = self.model_rootpath + target_model_folder_name + "/"
                    if not os.path.exists(target_model_folder_path):
                        continue
                    if not self.checkKeyExists(target_model_json, target_label_key):
                        continue
                    full_target_label = target_model_json[target_label_key]
                    if "/" not in full_target_label:
                        continue
                    target_label = full_target_label.split("/")[0]

                    have_this_label = False
                    for furniture in furniture_list:
                        if furniture[0] == target_label:
                            furniture[1].append(target_model_folder_path)
                            have_this_label = True
                            break
                    if not have_this_label:
                        new_furniture = [target_label, [target_model_folder_path]]
                        furniture_list.append(new_furniture)

                print("\rDataset detected : JsonFile : " + str(detected_source_json_num) + "/" + str(len(source_json_file_name_list)) + "    ", end="")
                if detected_source_json_num > 400:
                    print()
                    return furniture_list
        print()
        return furniture_list

    def createManifold40Dataset(self, model_class_list, output_dataset_path):
        if output_dataset_path[-1] != "/":
            output_dataset_path += "/"

        if os.path.exists(output_dataset_path):
            shutil.rmtree(output_dataset_path)
        os.makedirs(output_dataset_path)

        copyed_model_class_num = 0

        for model_class in model_class_list:
            copyed_model_class_num += 1
            model_label = model_class[0]
            model_path_list = model_class[1]
            output_model_folder_path = output_dataset_path + model_label + "/train/"

            if os.path.exists(output_model_folder_path):
                shutil.rmtree(output_model_folder_path)
            os.makedirs(output_model_folder_path)

            copyed_model_num = 0

            for model_folder_path in model_path_list:
                copyed_model_num += 1
                normalized_model_path = model_folder_path + "normalized_model.obj"
                raw_model_path = model_folder_path + "raw_model.obj"
                model_mtl_path = model_folder_path + "model.mtl"
                output_model_path = output_model_folder_path + model_label + "_" + str(copyed_model_num) + ".obj"

                source_model_path = normalized_model_path
                if not os.path.exists(source_model_path):
                    continue

                lines = None
                with open(source_model_path, "r") as fr:
                    lines = fr.readlines()
                with open(output_model_path, "w") as fw:
                    for line in lines:
                        if line[0] in ["v", "f"]:
                            fw.write(line)

                print("\rCreated : Model class : " + str(copyed_model_class_num) + "/" + str(len(model_class_list)) +
                      ", Model num : " + str(copyed_model_num) + "/" + str(len(model_path_list)) + "    ", end="")
        print()

if __name__ == '__main__':
    dataset_path = "/home/chli/3D_FRONT/3D-FRONT/3D-FRONT/"
    model_rootpath = "/home/chli/3D_FRONT/3D-FUTURE-model/3D-FUTURE-model/"
    output_dataset_path = "/home/chli/3D_FRONT/output/"

    td_front_dataset_loader = TDFRONTDatasetLoader()

    td_front_dataset_loader.loadDatasetPath(dataset_path, model_rootpath)

    furniture_list = td_front_dataset_loader.getFurnitureList()

    for i in range(len(furniture_list)):
        print("id: " + str(i) + ", label: " + furniture_list[i][0], ", Number = ", len(furniture_list[i][1]))

    td_front_dataset_loader.createManifold40Dataset(furniture_list, output_dataset_path)

