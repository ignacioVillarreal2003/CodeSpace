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
    content: 'IA_and_Machine_Learning_for_Coders/Chapter_1.md'
  },
  {
    id: 'unit-002',
    topicId: "top-001",
    title: 'Introducción a la visión artificial',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Chapter_2.md'
  },
  {
    id: 'unit-003',
    topicId: "top-001",
    title: 'Más Allá de lo Básico: Detectando Características en Imágenes',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Chapter_3.md'
  },
  {
    id: 'unit-004',
    topicId: "top-001",
    title: 'Usar conjuntos de datos públicos con TensorFlow Datasets',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Chapter_4.md'
  },
  {
    id: 'unit-005',
    topicId: "top-001",
    title: 'Introducción al Procesamiento del Lenguaje Natural',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Chapter_5.md'
  },
  {
    id: 'unit-006',
    topicId: "top-001",
    title: 'Haciendo que el sentimiento sea programable usando incrustaciones',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Chapter_6.md'
  },
];