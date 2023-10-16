import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    allowed_domains = ["quotes.toscrape.com"]
    start_urls = ["http://quotes.toscrape.com/"]


    # Nombre de archivo
    contador_txt = 0
    enlaces = []
    enlacesNuevos = []
    
    
    def parse(self, response):
        ''' Función para parsear una página '''
        # Obtener el texto de todas las etiquetas <p>
        parrafos = response.css('div::text, span::text, small::text, p::text, b::text, a::text').getall()
        # Obtener enlaces
        self.enlaces = response.css('a::attr(href)').getall()
        
        # Imprimir o guardar los párrafos según tus necesidades
        for parrafo in parrafos:
            print(parrafo)
        
        # Guardar pagina principal en un archivo
        nombre_txt = 'Doc' + str(self.contador_txt) + '.txt'
        
        with open(nombre_txt, 'w', encoding='utf-8') as archivo:
            for parrafo in parrafos:
                archivo.write(parrafo + ' ')
        self.contador_txt += 1
        
        # Guardar enlaces en un archivo
        with open('Enlaces.txt', 'w', encoding='utf-8') as archivo:
            for enlace in self.enlaces:
                archivo.write(enlace + '\n')

        self.contador_txt += 1
        
        input()         
        
        # Crawling profundo
        for _ in range(0, 1):
            # Sigue enlaces a n páginas
            for enlace in self.enlaces:
                yield response.follow(enlace, self.parse_pagina)
                
            self.enlaces = self.enlacesNuevos
            self.enlacesNuevos = []
            
                
    # Función para parsear una página
    def parse_pagina(self, response):        
        # Obtener el texto de todas las etiquetas <p>
        parrafos = response.css('div::text, span::text, small::text, p::text, b::text, a::text').getall()
        # Obtener enlaces
        enlaces = response.css('a::attr(href)').getall()

        # Imprimir o guardar los párrafos según tus necesidades
        for parrafo in parrafos:
            print(parrafo)
        
        nombre_txt = 'Doc' + str(self.contador_txt) + '.txt'
        
        # Guardar los párrafos en un archivo
        with open(nombre_txt, 'w', encoding='utf-8') as archivo:
            for parrafo in parrafos:
                archivo.write(parrafo + ' ')
        
        # Concatenar enlaces
        self.enlacesNuevos += enlaces
        # Incrementamos contador de txt
        self.contador_txt += 1
