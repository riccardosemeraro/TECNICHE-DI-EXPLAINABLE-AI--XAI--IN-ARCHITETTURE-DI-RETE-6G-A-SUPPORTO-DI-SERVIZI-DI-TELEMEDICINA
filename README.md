# TECNICHE DI EXPLAINABLE AI (XAI) IN ARCHITETTURE DI RETE 6G A SUPPORTO DI SERVIZI DI TELEMEDICINA
Progetto Tesi LT in Reti di Telecomunicazioni - TECNICHE DI EXPLAINABLE AI (XAI) IN ARCHITETTURE DI RETE 6G A SUPPORTO DI SERVIZI DI TELEMEDICINA

## ABSTRACT
Questo lavoro di tesi approfondisce l'integrazione tra la Rete 6G e eXplainable Artificial Intelligence (XAI), studiandone lo stato dell’arte di entrambi. Si sperimenta l’applicazione di XAI per l’analisi predittiva e la riduzione di dati e risorse di rete tramite un dataset biomedicale.

La sperimentazione si sviluppa in tre fasi: nella prima, i modelli di Random Forest (RF) sono addestrati sull'intero dataset; nella seconda, si utilizzano solo le feature più rilevanti individuate tramite SHapley Additive exPlanations (SHAP) e si addestra un nuovo modello con le sole feature più rilevanti; infine, nella terza fase, si applica l'Analisi delle Componenti Principali, ossia Principal Component Analysis (PCA), per ridurre la dimensionalità con l'individuazione delle componenti principali e si valuta l’efficienza di un nuovo training.

Per confrontare i diversi approcci, vengono esaminati i Key Performance Indicators (KPIs) rilevanti per le Reti 6G, con l’obiettivo di collegare le capacità predittive dei modelli alle esigenze prestazionali di questa nuova infrastruttura tecnologica.


## INTRODUZIONE
Negli ultimi anni, il settore delle Reti di Telecomunicazioni ha visto una crescita esponenziale con l’evoluzione delle reti mobili, culminando nello sviluppo del 5G. Tuttavia, l’attenzione si è rapidamente spostata verso la prossima generazione di infrastrutture: le reti 6G, progettate per garantire una connettività globale e ubiqua, riducendo al minimo la latenza e supportando applicazioni critiche come la telemedicina avanzata, l’Internet of Things (IoT) e la realtà aumentata/virtuale. In questo contesto, l’integrazione con tecnologie di Artificial Intelligence (AI) diventa cruciale, poiché la gestione automatica e intelligente delle risorse di rete migliorerà efficienza, flessibilità e capacità di auto-ottimizzazione. Tuttavia, l’adozione su larga scala dell’AI pone sfide significative, soprattutto riguardo alla trasparenza e alla fiducia nelle decisioni prese dai modelli di apprendimento automatico, noti per il fenomeno della "black-box", ovvero l’opacità nelle loro logiche interne.

Proprio per affrontare questo genere di problemi, si stanno sviluppando tecniche di eXplainable Artificial Intelligence (XAI). Le tecniche di XAI si pongono l'obiettivo di trasformare i modelli di intelligenza artificiale, spesso percepiti, come detto in precedenza, come delle "black-box" per la loro complessità e opacità, in "white-box" in cui le decisioni possono essere comprese e giustificate. Queste metodologie rendono i modelli di AI più comprensibili e interpretabili, consentendo a operatori di rete, utenti e autorità di regolamentazione di analizzare e verificare le decisioni prese dai sistemi di AI. In un contesto di rete come il 6G, dove le applicazioni critiche dipendono da decisioni rapide e accurate, la spiegabilità delle decisioni diventa essenziale per garantire affidabilità, sicurezza e conformità alle normative sulla privacy.

Questo lavoro di tesi esplora l’impiego delle tecniche di XAI nel contesto delle reti 6G, focalizzandosi sulla loro applicazione in ambito della telemedicina. La telemedicina è uno dei casi d’uso più promettenti delle reti 6G, in cui la bassa latenza e la connettività stabile permettono lo svolgimento di operazioni a distanza e il monitoraggio continuo dei pazienti tramite dispositivi biometrici e wearable. Tuttavia, la gestione di grandi quantità di dati clinici in tempo reale richiede soluzioni efficienti per ridurre il carico sulle risorse di rete e garantire tempi di risposta rapidi. In questo contesto, XAI non solo migliora la trasparenza delle predizioni mediche, ma consente anche di ottimizzare l’allocazione delle risorse.

L’obiettivo principale dello studio condotto è valutare come l’uso delle tecniche XAI possa migliorare l'efficienza e la comprensibilità delle decisioni all'interno delle architetture di rete 6G. In particolare, il lavoro sperimentale condotto in questa tesi si basa sull'utilizzo di un dataset biomedicale per testare l'efficacia delle tecniche di XAI per la predizione di condizioni mediche. Il processo è suddiviso in tre fasi principali: inizialmente, un modello di Random Forest (RF) verrà addestrato sull’intero dataset; successivamente, verrà applicata la tecnica SHapley Additive exPlanations (SHAP) per individuare le feature più rilevanti; infine, si confronteranno i ristati ottenuti con la Principal Component Analysis (PCA), tecnica standard per ridurre la dimensionalità dei dati. Lo studio permetterà di valutare come la riduzione dei dati influisca sull'accuratezza del modello e sull'efficienza di trasmissione in termini di risparmio di risorse di rete.

Il confronto tra questi approcci verrà condotto analizzando vari Key Performance Indicators (KPI) rilevanti per le reti 6G, con l’obiettivo di collegare la capacità predittiva dei modelli di AI alle esigenze prestazionali della nuova infrastruttura tecnologica, cercando di offrire soluzioni che bilancino l'efficacia delle predizioni con la sostenibilità e l'efficienza delle risorse di rete. 

L'analisi tramite KPI consentirà di valutare come la riduzione del carico dati possa tradursi in un miglioramento significativo delle prestazioni di rete, senza compromettere eccessivamente la qualità delle predizioni mediche. Questo bilanciamento è essenziale per garantire che i servizi di telemedicina possano essere affidabili e sostenibili, anche in contesti con limitate risorse energetiche o di rete.

Il presente lavoro di tesi si inserisce, dunque, in un contesto di ricerca interdisciplinare, che mira a coniugare l'efficienza tecnologica con la trasparenza e l'affidabilità necessarie per applicazioni critiche come la telemedicina.


## CONCLUSIONI
L'analisi condotta in questo lavoro di tesi ha dimostrato come l’integrazione di tecniche di XAI nelle Reti 6G possa migliorare significativamente la trasparenza e l’efficienza operativa dei modelli di AI, rendendoli adatti a contesti critici come la telemedicina. Nello studio con tutto il dataset biomedicale relativo al diabete, preso in esame nella tesi, il modello RF utilizzato ha mostrato un’elevata accuratezza predittiva, pari al 99%. Però, l'analisi del dataset completo ha rivelato limiti in termini di consumo energetico e risorse di rete. Per superare tali ostacoli, sono state  ridotte le feature tramite SHAP e PCA, che hanno consentito un risparmio significativo in termini di tempo di trasmissione ed energia, mantenendo una buona accuratezza.

I risultati mostrano come, rispetto all'addestramento del modello di RF sul dataset completo, la selezione delle prime tre feature più importanti abbia comportato una riduzione del 55,56% nel tempo di trasmissione e nell'energia impiegata, mantenendo un livello di accuratezza soddisfacente di circa il 95%. Un ulteriore caso analizzato è stato ottenuto con l’utilizzo delle prime due componenti principali tramite PCA, che hanno garantito un’accuratezza del 97% e un risparmio del 64,49%, dimostrando così che è possibile ottenere un equilibrio tra accuratezza predittiva e sostenibilità energetica nelle reti 6G, senza però per permettere la spiegabilità. Complessivamente, queste ottimizzazioni sul numero di colonne del dataset rendono l'integrazione di XAI particolarmente vantaggiosa per scenari come la telemedicina, migliorando la qualità del servizio senza compromettere l’efficienza operativa. 

Dal punto di vista applicativo, quindi, l'adozione di XAI nelle Reti 6G non solo migliora la trasparenza delle decisioni prese dall’AI, ma favorisce anche l’accettazione di queste tecnologie da parte degli operatori e degli utenti finali, incrementando la fiducia e la conformità normativa. La possibilità di spiegare le decisioni è particolarmente importante in ambito medico, dove le predizioni basate sull’AI possono influenzare direttamente la vita dei pazienti. Grazie alle tecniche di XAI, i medici possono comprendere le motivazioni alla base delle diagnosi e dei trattamenti suggeriti dai modelli di AI, garantendo così un processo decisionale più consapevole e sicuro.

In futuro, sarà essenziale esplorare l’uso di modelli di \gls{ai} più avanzati, come le DNN, integrando tecniche di XAI per spiegare anche le decisioni più complesse. Inoltre, la combinazione di XAI con PCA con il paradigma di Edge Computing potrebbe rappresentare un ulteriore progresso, consentendo di effettuare analisi complesse ai margini della rete, riducendo così la latenza e migliorando l’efficienza complessiva del sistema. Un’altra direzione di ricerca interessante sarà l’analisi dell'impatto della riduzione del carico dati sui costi operativi delle reti, in particolare in contesti di telemedicina e in applicazioni reali delle Smart City e della gestione ambientale.

Sperimentare con modelli più sofisticati e con la combinazione di approcci di XAI potrebbe portare a miglioramenti significativi nella qualità e nell'affidabilità delle decisioni prese dalle reti 6G, creando infrastrutture di comunicazione più human-centric e trasparenti. Infine, la collaborazione tra industria, accademia e istituzioni sarà fondamentale per garantire che le Reti 6G evolvano non solo come infrastrutture tecnologiche, ma anche come strumenti etici e socialmente responsabili.

In conclusione, il lavoro di tesi ha dimostrato che l’integrazione di XAI e 6G rappresenta una strada promettente per il futuro delle reti di telecomunicazioni tramite un caso di studio dei servizi di telemedicina. Lo sviluppo di infrastrutture di rete più trasparenti, intelligenti e sostenibili sarà essenziale per affrontare le sfide poste da un mondo sempre più connesso, garantendo al tempo stesso la sicurezza, l’efficacia, la trasparenza e la fiducia degli utenti.


## TERMINI E CONDIZIONI D'USO

<p align="center">
© [TECNICHE DI EXPLAINABLE AI (XAI) IN ARCHITETTURE DI RETE 6G A SUPPORTO DI SERVIZI DI TELEMEDICINA] - [2023/2024] <br>
  <br>
<strong>IL CODICE È FORNITO “COSÌ COM’È” SENZA GARANZIA DI ALCUN TIPO, CONCESSO A TITOLO GRATUITO A QUALSIASI PERSONA E DI UTILIZZARLO SENZA RESTRIZIONI, ESCLUDENDO FINI DI COMMERCIABILITÀ. IN NESSUN CASO GLI AUTORI DEL CODICE SARANNO RESPONSABILI PER QUALSIASI RECLAMO, DANNI O ALTRA RESPONSABILITÀ, DERIVANTI DA O IN CONNESSIONE CON IL CODICE O L’USO.</strong><br>
<br>
L’avviso di copyright sopra riportato e questo avviso di permesso devono essere inclusi in tutte le copie o porzioni sostanziali, citando gli autori del suddetto: <br>
<br>
Lavoro di Tesi Studiato e Realizzato da <br>
  <br>
Riccardo Semeraro - <a href="https://github.com/riccardosemeraro">Link GitHub</a> <br>
<br>
Uno Studente del <a href="http://www.poliba.it/">Politecnico di Bari</a> <br>
<br>

</p>
