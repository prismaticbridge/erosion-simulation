// mod bridge_data;
mod terrain;
mod test_terrain;
use winit::error::EventLoopError;
use winit::event_loop::EventLoop;

use crate::test_terrain::App;

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
