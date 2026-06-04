import matplotlib.pyplot as plt
import pandas as pd
from dataset import NERDatasetLoader

def compare_datasets(dataset_names):
    """比较不同数据集的统计特征"""
    stats_list = []
    
    for name in dataset_names:
        print(f"\n分析数据集: {name}")
        loader = NERDatasetLoader(name)
        
        try:
            train_data = loader.load_data('train')
            config = loader.config
            
            # 计算统计信息
            total_tokens = sum(len(sample['tokens']) for sample in train_data)
            total_entities = 0
            entity_counts = {}
            
            for sample in train_data:
                for tag in sample['ner_tags']:
                    if tag != 'O':
                        total_entities += 1
                        prefix = tag.split('-')[0]  # B or I
                        if prefix == 'B':
                            entity_type = tag[2:]
                            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            stats = {
                '数据集': name,
                '样本数': len(train_data),
                '总字符数': total_tokens,
                '总实体数': total_entities,
                '实体类型数': len(config.entity_types),
                '实体密度': total_entities / total_tokens * 1000,  # 每千字符实体数
                '主要实体类型': ', '.join(list(entity_counts.keys())[:3])
            }
            stats_list.append(stats)
            
        except Exception as e:
            print(f"  错误: {e}")
    
    # 创建比较表格
    df = pd.DataFrame(stats_list)
    print("\n" + "="*80)
    print("数据集对比统计")
    print("="*80)
    print(df.to_string(index=False))
    
    return df
