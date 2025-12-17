#!/usr/bin/env python3
"""
本体管理器
负责加载 RDF 本体结构，并从 CSV 文件填充具体的传感器、算法和基础设施数据。
"""

import logging
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)

# 定义命名空间
RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class OntologyManager:
    """管理 RDTcO-Maint 本体操作"""

    def __init__(self):
        self.g = Graph()
        self.g.bind("rdtco", RDTCO)
        self.g.bind("ex", EX)
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("owl", OWL)
        self.g.bind("xsd", XSD)

        # 初始化基础结构
        self._setup_base_ontology()

    def _setup_base_ontology(self):
        """设置基础本体类结构"""
        # 简单的类定义，确保后续实例化不会报错
        classes = [
            'SensorSystem', 'Algorithm', 'StorageSystem',
            'CommunicationSystem', 'ComputeDeployment',
            'MMS_LiDAR_System', 'MMS_Camera_System', 'UAV_LiDAR_System',
            'UAV_Camera_System', 'TLS_System', 'Handheld_3D_Scanner',
            'FiberOptic_Sensor', 'Vehicle_LowCost_Sensor', 'IoT_Network_System'
        ]

        for cls in classes:
            self.g.add((RDTCO[cls], RDF.type, OWL.Class))

    def populate_from_csv_files(self, sensor_csv, algorithm_csv, infrastructure_csv, cost_benefit_csv):
        """从 CSV 文件填充本体图"""
        logger.info("正在从 CSV 文件填充本体...")

        try:
            self._load_sensors(sensor_csv)
            self._load_algorithms(algorithm_csv)
            self._load_infrastructure(infrastructure_csv)
            self._load_cost_benefit(cost_benefit_csv)
            logger.info(f"本体填充完成，包含 {len(self.g)} 个三元组。")
            return self.g
        except Exception as e:
            logger.error(f"填充本体时出错: {e}")
            raise

    def _load_sensors(self, filepath):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            uri = EX[self._clean_id(row['Sensor_Instance_Name'])]
            rdftype = RDTCO[row['Sensor_RDF_Type']]
            self.g.add((uri, RDF.type, rdftype))

            # 添加属性
            self._add_prop(uri, 'hasInitialCostUSD', row.get('Initial_Cost_USD'))
            self._add_prop(uri, 'hasOperationalCostUSDPerDay', row.get('Operational_Cost_USD_per_day'))
            self._add_prop(uri, 'hasEnergyConsumptionW', row.get('Energy_Consumption_W'))
            self._add_prop(uri, 'hasMTBFHours', row.get('MTBF_hours'))
            self._add_prop(uri, 'hasCoverageEfficiencyKmPerDay', row.get('Coverage_Efficiency_km_per_day'))
            self._add_prop(uri, 'hasAccuracyRangeMM', row.get('Accuracy_Range_mm'))
            self._add_prop(uri, 'hasDataVolumeGBPerKm', row.get('Data_Volume_GB_per_km'))
            self._add_prop(uri, 'hasOperatingSpeedKmh', row.get('Operating_Speed_kmh'))
            self._add_prop(uri, 'hasOperatorSkillLevel', row.get('Operator_Skill_Level'), is_literal=True)

    def _load_algorithms(self, filepath):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            uri = EX[self._clean_id(row['Algorithm_Instance_Name'])]
            rdftype = RDTCO[row['Algorithm_RDF_Type']]
            self.g.add((uri, RDF.type, rdftype))

            self._add_prop(uri, 'hasRecall', row.get('Recall'))
            self._add_prop(uri, 'hasPrecision', row.get('Precision'))
            self._add_prop(uri, 'hasDataAnnotationCostUSD', row.get('Data_Annotation_Cost_USD'))
            self._add_prop(uri, 'hasModelRetrainingFreqMonths', row.get('Model_Retraining_Freq_months'))
            self._add_prop(uri, 'hasHardwareRequirement', row.get('Hardware_Requirement'), is_literal=True)
            self._add_prop(uri, 'hasExplainabilityScore', row.get('Explainability_Score'))

    def _load_infrastructure(self, filepath):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            uri = EX[self._clean_id(row['Component_Instance_Name'])]
            rdftype = RDTCO[row['Component_RDF_Type']]
            self.g.add((uri, RDF.type, rdftype))

            self._add_prop(uri, 'hasInitialCostUSD', row.get('Initial_Cost_USD'))
            self._add_prop(uri, 'hasAnnualOpCostUSD', row.get('Annual_OpCost_USD'))
            self._add_prop(uri, 'hasEnergyConsumptionW', row.get('Energy_Consumption_W'))
            self._add_prop(uri, 'hasMTBFHours', row.get('MTBF_hours'))

    def _load_cost_benefit(self, filepath):
        # 简单加载，不做复杂处理
        pass

    def _add_prop(self, subject, pred_name, value, is_literal=False):
        """辅助函数：添加属性"""
        if pd.isna(value) or str(value) == 'N/A':
            return

        predicate = RDTCO[pred_name]
        if is_literal:
            self.g.add((subject, predicate, Literal(str(value))))
        else:
            try:
                # 尝试转为数字
                val = float(value)
                self.g.add((subject, predicate, Literal(val, datatype=XSD.decimal)))
            except:
                self.g.add((subject, predicate, Literal(str(value))))

    def _clean_id(self, text):
        """清理字符串以用作 URI"""
        return str(text).replace(" ", "_").replace("(", "").replace(")", "").strip()