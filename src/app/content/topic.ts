export interface Topic {
  id: string,
  categoryId: string,
  title: string,
  color: string
}

export const topics: Topic[] = [
  {
    id: 'top-001',
    categoryId: "cat-001",
    title: 'Inteligencia artificial y aprendizaje automático para programadores',
    color: "#f46464"
  }
];
