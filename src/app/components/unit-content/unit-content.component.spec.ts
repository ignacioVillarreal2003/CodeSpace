import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UnitContentComponent } from './unit-content.component';

describe('TopicContentComponent', () => {
  let component: UnitContentComponent;
  let fixture: ComponentFixture<UnitContentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UnitContentComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UnitContentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
