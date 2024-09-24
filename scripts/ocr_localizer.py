import easyocr

class OCRLocalizer:
    def __init__(self,img_path,debug) -> None:
        self.reader = easyocr.Reader(['en'])
        self.result = self.reader.readtext(img_path)
        self.found_tokens = {}

        self.oil_dict = ["OIL","OI","IL","O"]
        self.vinegar_dict = ["V","G","A","R",
                             "VI","IN","NE","EG","GA","AR",
                             "VIN","INE","NEG","EGA","GAR",
                             "VINE","INEG","NEGA","EGAR",
                             "VINEG","INEGA","NEGAR",
                             "VINEGA","INEGAR",
                             "VINEGAR"]
        
        # Get found tokens from result
        for (bbox, text, prob) in self.result:
            self.found_tokens[text] = prob
            #print(f'Text: {text}, Probability: {prob}')
        print("Found tokens: "+str(self.found_tokens.keys()))

        # Status
        self.status = "FAIL"
        p_thresh = 0.1

        # Check if oil token was found
        for ot in self.oil_dict:
            for ft in self.found_tokens.keys():
                ftl = ft.lower()
                otl = ot.lower()
                if ftl == otl or ftl in otl or otl in ftl:
                    p = self.found_tokens[ft]
                    if p > p_thresh:
                        if debug:
                            print("Oil token found: "+str(ft)+" with p="+"%.2f" % p)
                        self.status = "oil"
                        return
        
        # If oil token was not found check if vinegar token was found
        if self.status == "FAIL":
            for vt in self.vinegar_dict:
                for ft in self.found_tokens.keys():
                    vtl = vt.lower()
                    ftl = ft.lower()
                    if vtl == ftl or ftl in vtl or vtl in ftl:
                        p = self.found_tokens[ft]
                        if p > p_thresh:
                            if debug:
                                print("Vinegar token found: "+str(ft)+" with p="+"%.2f" % p)
                            self.status = "vinegar"
                            return

