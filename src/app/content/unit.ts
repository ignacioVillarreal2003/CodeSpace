export interface Unit {
  id: string,
  topicId: string,
  title: string,
  color: string,
  content: string
}

export const units: Unit[] = [
  {
    id: 'unit-001',
    topicId: "top-001",
    title: 'Introducción a TensorFlow',
    color: "#438ccf",
    content: 'IA_and_Machine_Learning_for_Coders/Introduction to TensorFlow.md'
  },
  {
    id: 'unit-002',
    topicId: "top-001",
    title: 'Introducción a la visión artificial',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Introduction to Computer Vision.md'
  },
  {
    id: 'unit-003',
    topicId: "top-001",
    title: 'Más Allá de lo Básico: Detectando Características en Imágenes',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Going_Beyond_the_Basics_Detecting_Features_in_Images.md'
  },
  {
    id: 'unit-004',
    topicId: "top-001",
    title: 'Usar conjuntos de datos públicos con TensorFlow Datasets',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Using_Public_Datasets_with_TensorFlow_Datasets.md'
  },
  {
    id: 'unit-005',
    topicId: "top-001",
    title: 'Introducción al Procesamiento del Lenguaje Natural',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Introduction_to_Natural_Language_Processing.md'
  },
];