import { Container, Text, Button, Group } from '@mantine/core';
// import { GithubIcon } from '@mantinex/dev-icons';
import classes from './landing.module.css';
import FeaturesGrid from '../FeatureGrid';

export default function LandingSection() {
  return (
    <div className={classes.wrapper}>
      <Container size={700} className={classes.inner}>
        <h1 className={classes.title}>
          A{' '}
          <Text
            component="span"
            variant="gradient"
            gradient={{ from: 'blue', to: 'cyan' }}
            inherit
          >
            fully featured
          </Text>{' '}
          React components and hooks library
        </h1>

        <Text className={classes.description} color="dimmed">
          Build fully functional accessible web applications with ease – Mantine
          includes more than 100 customizable components and hooks to cover you
          in any situation
        </Text>

        <Group className={classes.controls}>
          <Button
            size="xl"
            className={classes.control}
            variant="gradient"
            gradient={{ from: 'blue', to: 'cyan' }}
          >
            Get started
          </Button>

          <Button
            component="a"
            href="https://github.com/kookmin-sw/capstone-2024-05"
            size="xl"
            variant="default"
            className={classes.control}
            target="_blank"
            // leftSection={<GithubIcon size={20} />}
          >
            GitHub
          </Button>
        </Group>
      </Container>
      <Container size={700} className={classes.inner}>
        <FeaturesGrid />
      </Container>
    </div>
  );
}
