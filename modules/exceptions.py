class TooManyFaces(Exception):
    def __init__(self,
                 message="Muitos rostos detectados na imagem. Por favor, forneça uma imagem que contenha exatamente um rosto."):
        self.message = message
        super().__init__(self.message)


class NoFaces(Exception):
    def __init__(self,
                 message="Nenhum rosto foi detectado na imagem. Por favor, forneça uma imagem que contenha um rosto."):
        self.message = message
        super().__init__(self.message)
