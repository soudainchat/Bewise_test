import transformers


class Ner_Extractor:
    
    def __init__(self, model_checkpoint: str):
        self.token_pred_pipeline = transformers.pipeline("token-classification", 
                                            model=model_checkpoint, 
                                            aggregation_strategy="average")
    
    @staticmethod
    def concat_entities(ner_result):

        entities = []
        prev_entity = None
        prev_end = 0
        for i in range(len(ner_result)):
            
            if (ner_result[i]["entity_group"] == prev_entity) &\
               (ner_result[i]["start"] == prev_end):
                
                entities[i-1][2] = ner_result[i]["end"]
                prev_entity = ner_result[i]["entity_group"]
                prev_end = ner_result[i]["end"]
            else:
                entities.append([ner_result[i]["entity_group"], 
                                 ner_result[i]["start"], 
                                 ner_result[i]["end"]])
                prev_entity = ner_result[i]["entity_group"]
                prev_end = ner_result[i]["end"]
        
        return entities

    def get_entities(self, text: str):

        assert len(text) > 0, text
        entities = self.token_pred_pipeline(text)
        concat_ent = self.concat_entities(entities)
        words = [(concat_ent[i][0],text[concat_ent[i][1]:concat_ent[i][2]]) for i in range(len(concat_ent))]
        
        return words